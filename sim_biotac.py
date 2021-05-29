# Copyright (c) 2021 NVIDIA Corporation

# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
# OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
# ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.

"""
Simulate indentations of the BioTac sensor with 8 different indenters for a sample trajectory.

Extracts net force vectors, nodal coordinates, and optionally, element-wise stresses for the BioTac.
"""

import argparse
import copy
import fileinput
import h5py
import numpy as np
import os
from scipy.spatial.transform import Rotation as R
from shutil import copyfile

from carbongym import gymapi
from carbongym import gymtorch
from carbongym import gymutil

# Set target indenter pose
INDENT_TARGET = [-0.015, 0.0425, 0.0,   # Position of tip
                 1.0, 0.0, 0.0,         # Orientation of x-axis       
                 0.0, 1.0, 0.0,         # Orientation of y-axis
                 0.0, 0.0, 1.0]         # Orientation of z-axis

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--elast_mod', default=1.55223093e+06, type=float, help="Elastic modulus of BioTac [Pa]")
    parser.add_argument('--poiss_ratio', default=3.16454280e-01, type=float, help="Poisson's ratio of BioTac")
    parser.add_argument('--frict_coeff', default=7.83414394e-01, type=float, help='Coefficient of friction between BioTac and indenter')
    parser.add_argument('--indent_dist', default=5.0e-3, type=float, help='Initial distance of indenter [m] from target point on BioTac')
    parser.add_argument('--indent_steps', default=50, type=int, help='Number of indentation steps (i.e., spatial increments) over which to collect data')
    parser.add_argument('--err_tol', default=1.0e-4, type=float, help='Maximum acceptable error [m] between target and actual position of indenter')
    parser.add_argument('--extract_stress', default=False, type=bool, help='Extract stress at each indentation step (will reduce simulation speed)')
    parser.add_argument('--export_results', default=True, type=bool, help='Export results to HDF')
    args = parser.parse_args()

    gym = gymapi.acquire_gym()

    # Define sim parameters and create Sim object
    sim = create_sim(gym=gym,
                     indent_dist=args.indent_dist,
                     indent_steps=args.indent_steps,
                     frict_coeff=args.frict_coeff)

    # Define and load assets
    asset_options = set_asset_options()
    biotac_urdf_dir = os.path.join('urdf', 'biotac', 'shell_core')
    set_biotac_matl_props(base_dir=biotac_urdf_dir,
                          elast_mod=args.elast_mod,
                          poiss_ratio=args.poiss_ratio)
    asset_handles_biotac = load_assets(gym=gym,
                                       sim=sim,
                                       base_dir=biotac_urdf_dir,
                                       objects=['biotac'],
                                       options=asset_options)
    indenter_names = ['sphere_3-5mm', 'sphere_7mm', 'sphere_14mm', 
                      'cylinder_short_3-5mm', 'cylinder_short_7mm', 'cylinder_long_7mm', 
                      'cube_14mm', 'ring_7mm']
    asset_handles_indenters = load_assets(gym=gym,
                                          sim=sim,
                                          base_dir=os.path.join('urdf', 'indenters'),
                                          objects=indenter_names,
                                          options=asset_options)

    # Define and create scene
    scene_props = set_scene_props(num_envs=len(indenter_names))
    env_handles, actor_handles_biotacs, actor_handles_indenters = create_scene(gym=gym, 
                                                                               sim=sim, 
                                                                               props=scene_props,
                                                                               assets_biotac=asset_handles_biotac,
                                                                               assets_indenters=asset_handles_indenters)    
    viewer, axes_geom = create_viewer(gym=gym, 
                                      sim=sim)

    # Define controller for indenters
    set_ctrl_props(gym=gym,
                   envs=env_handles,
                   indenters=actor_handles_indenters)

    # Run simulation loop
    results = run_sim_loop(gym=gym, 
                           sim=sim, 
                           envs=env_handles, 
                           indenters=actor_handles_indenters,
                           viewer=viewer,
                           axes=axes_geom,
                           indent_target=INDENT_TARGET,
                           indent_dist=args.indent_dist,
                           indent_steps=args.indent_steps,
                           err_tol=args.err_tol,
                           extract_stress=args.extract_stress)

    # Export results to HDF
    if args.export_results:
        export_results(results, path='results.hdf5')

def create_scene(gym, sim, props, assets_biotac, assets_indenters, biotac_offset=0.05, indenter_offset=0.01785938):
    """Create a scene (i.e., ground plane, environments, BioTac actors, and indenter actors)."""

    plane_params = gymapi.PlaneParams()
    gym.add_ground(sim, plane_params)

    env_handles = []
    actor_handles_biotacs = []
    actor_handles_indenters = []
    for i in range(props['num_envs']):
        env_handle = gym.create_env(sim, props['lower'], props['upper'], props['per_row'])
        env_handles.append(env_handle)

        pose = gymapi.Transform()
        collision_group = i
        collision_filter = 0

        pose.p = gymapi.Vec3(0.0, biotac_offset, 0.0)
        r = R.from_euler('XYZ', [0, 0, 0], degrees=True)
        quat = r.as_quat()
        pose.r = gymapi.Quat(*quat)
        actor_handle_biotac = gym.create_actor(env_handle, assets_biotac[0], pose, f"biotac_{i}", collision_group, collision_filter)
        actor_handles_biotacs.append(actor_handle_biotac)

        pose.p = gymapi.Vec3(0.0, indenter_offset, 0.0)
        r = R.from_euler('XYZ', [0, 0, 0], degrees=True)
        quat = r.as_quat()
        pose.r = gymapi.Quat(*quat)
        actor_handle_indenter = gym.create_actor(env_handle, assets_indenters[i], pose, f"indenter_{i}", collision_group, collision_filter)
        actor_handles_indenters.append(actor_handle_indenter)

    return env_handles, actor_handles_biotacs, actor_handles_indenters

def create_sim(gym, indent_dist, indent_steps, frict_coeff):
    """Set the simulation parameters and create a Sim object."""

    sim_type = gymapi.SIM_FLEX
    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0e-4  # Control frequency
    sim_params.substeps = 1  # Physics simulation frequency (multiplier)
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, 0.0)

    sim_params.stress_visualization = True  # von Mises stress
    sim_params.stress_visualization_min = 1.0e4
    sim_params.stress_visualization_max = 1.0e5

    sim_params.flex.solver_type = 5  # PCR (GPU, global)
    sim_params.flex.num_outer_iterations = 8
    sim_params.flex.num_inner_iterations = 40
    sim_params.flex.relaxation = 0.75
    sim_params.flex.warm_start = 0.8
    sim_params.flex.deterministic_mode = True

    sim_params.flex.geometric_stiffness = 1.0
    sim_params.flex.shape_collision_distance = indent_dist / indent_steps  # Distance to be maintained between soft bodies and other bodies or ground plane
    sim_params.flex.shape_collision_margin = indent_dist / indent_steps * 2  # Distance from rigid bodies at which to begin generating contact constraints

    sim_params.flex.friction_mode = 2  # Friction about all 3 axes (including torsional)
    sim_params.flex.dynamic_friction = frict_coeff

    gpu_physics = 0
    gpu_render = 0
    sim = gym.create_sim(gpu_physics, gpu_render, sim_type, sim_params)

    return sim

def create_viewer(gym, sim):
    """Create viewer and axes objects."""

    camera_props = gymapi.CameraProperties()
    camera_props.horizontal_fov = 5.0
    camera_props.width = 1920
    camera_props.height = 1080
    viewer = gym.create_viewer(sim, camera_props)
    camera_pos = gymapi.Vec3(0.075, 1.0, 4.0)
    camera_target = gymapi.Vec3(0.075, 0.0, 0.0)
    gym.viewer_camera_look_at(viewer, None, camera_pos, camera_target)

    axes_geom = gymutil.AxesGeometry(0.1)

    return viewer, axes_geom

def export_results(results, path):
    """Export the results to an HDF file."""

    # If previous HDF5 file exists, delete it
    if os.path.exists(path):
        os.remove(path)

    with h5py.File(path, 'w') as f:
        for i, step in enumerate(results):
            step_grp = f.create_group(f'step_{i}')
            for feature in step:
                feature_grp = step_grp.create_group(feature)
                for j, env in enumerate(step[feature]):
                    feature_grp.create_dataset(f'env_{j}', data=env)

def extract_results(gym, sim, envs, particle_states, extract_stress):
    """Extract the results from all environments."""

    results = {}
    results['net_force_vecs'] = extract_net_forces(gym=gym, 
                                                   sim=sim)  # (num_env x 3)
    results['nodal_coords'] = extract_nodal_coords(gym=gym, 
                                                   sim=sim, 
                                                   particle_states=particle_states)  # (num_env x num_nodes x 3)
    if extract_stress:
        results['elem_stresses'] = extract_elem_stresses(gym=gym,
                                                         sim=sim,
                                                         envs=envs)  # (num_env x num_elems)

    return results

def extract_elem_stresses(gym, sim, envs):
    """Extract the element-wise von Mises stresses on the BioTac from each environment."""

    (_, stresses) = gym.get_sim_tetrahedra(sim)
    num_envs = gym.get_env_count(sim)
    num_tets = len(stresses)
    num_tets_per_env = int(num_tets / num_envs)
    stresses_von_mises = np.zeros((num_envs, num_tets_per_env))

    for env_index, env in enumerate(envs):
        # Get tet range (start, count) for BioTac
        tet_range = gym.get_actor_tetrahedra_range(env, 0, 0)

        # Compute and store von Mises stress for each tet
        # TODO: Vectorize for speed
        for global_tet_index in range(tet_range.start, tet_range.start + tet_range.count):
            stress = stresses[global_tet_index]
            stress = np.matrix([(stress.x.x, stress.y.x, stress.z.x),
                                (stress.x.y, stress.y.y, stress.z.y),
                                (stress.x.z, stress.y.z, stress.z.z)])
            stress_von_mises = np.sqrt(0.5 * \
                                       ((stress[0, 0] - stress[1, 1]) ** 2 \
                                      + (stress[1, 1] - stress[2, 2]) ** 2 \
                                      + (stress[2, 2] - stress[0, 0]) ** 2 \
                                      + 6 * (stress[1, 2] ** 2 + stress[2, 0] ** 2 + stress[0, 1] ** 2)))
            local_tet_index = global_tet_index % num_tets_per_env
            stresses_von_mises[env_index][local_tet_index] = stress_von_mises
    
    return stresses_von_mises

def extract_net_forces(gym, sim):
    """Extract the net force vector on the BioTac for each environment."""

    contacts = gym.get_soft_contacts(sim)
    num_envs = gym.get_env_count(sim)
    net_force_vecs = np.zeros((num_envs, 3))
    for contact in contacts:
        rigid_body_index = contact[4]
        contact_normal = np.array([*contact[6]])
        contact_force_mag = contact[7]
        env_index = rigid_body_index // 3
        force_vec = contact_force_mag * contact_normal
        net_force_vecs[env_index] += force_vec
    net_force_vecs = -net_force_vecs
    
    return net_force_vecs

def extract_nodal_coords(gym, sim, particle_states):
    """Extract the nodal coordinates for the BioTac from each environment."""

    gym.refresh_particle_state_tensor(sim)
    num_envs = gym.get_env_count(sim)
    num_particles = len(particle_states)
    num_particles_per_env = int(num_particles /  num_envs)
    nodal_coords = np.zeros((num_envs, num_particles_per_env, 3))
    for global_particle_index, particle_state in enumerate(particle_states):
        pos = particle_state[:3]
        env_index = global_particle_index // num_particles_per_env
        local_particle_index = global_particle_index % num_particles_per_env
        nodal_coords[env_index][local_particle_index] = pos.numpy()
    
    return nodal_coords

def get_pose_and_draw(gym, env, viewer, axes, indenter):
    """Draw the pose of an indenter."""

    indenter_pose = gym.get_actor_rigid_body_states(env, indenter, gymapi.STATE_POS)['pose']
    pose_to_draw = gymapi.Transform.from_buffer(indenter_pose)
    gymutil.draw_lines(axes, gym, viewer, env, pose_to_draw)

def load_assets(gym, sim, base_dir, objects, options, fix=True, gravity=False):
    """Load assets from specified URDF files."""

    options.fix_base_link = True if fix else False
    options.disable_gravity = True if not gravity else False
    handles = []
    for obj in objects:
        handle = gym.load_asset(sim, base_dir, obj + '.urdf', options)
        handles.append(handle)
    
    return handles

def reset_biotac_state(gym, sim, biotac_state):
    """Reset the BioTac particle states to their initial states."""

    gym.set_particle_state_tensor(sim, gymtorch.unwrap_tensor(biotac_state))

def reset_indenter_state(gym, envs, indenters, indent_target, indent_dist):
    """Reset the indenter states to the initial state of a new trajectory."""

    indenter_index = 0
    for env, indenter in zip(envs, indenters):
        indenter_state = gym.get_actor_rigid_body_states(env, indenter, gymapi.STATE_ALL)

        target_pos = indent_target[:3]
        target_x_axis = indent_target[3:6]
        target_y_axis = indent_target[6:9]
        target_z_axis = indent_target[9:12]

        # Translate tip of indenter to target, minus indentation distance. Rotate indenter axes to target axes
        pos = np.asarray(target_pos) - (indent_dist) * np.asarray(target_y_axis)
        r = R.from_matrix(np.asarray([target_x_axis, target_y_axis, target_z_axis]).transpose())
        quat = r.as_quat()
        for link_state in indenter_state:
            link_state['pose']['p'] = tuple(pos)
            link_state['pose']['r'] = tuple(quat)
            link_state['vel']['linear'] = (0.0, 0.0, 0.0)
            link_state['vel']['angular'] = (0.0, 0.0, 0.0)
        gym.set_actor_rigid_body_states(env, indenter, indenter_state, gymapi.STATE_ALL)

        indenter_index += 1

def run_sim_loop(gym, sim, envs, indenters, viewer, axes, indent_target, indent_dist, indent_steps, err_tol, extract_stress, draw_pose=False, show_contacts=False):
    """Run the simulation and visualization loop."""

    results = []
    
    # Get particle state tensor and convert to PyTorch tensor
    particle_state_tensor = gymtorch.wrap_tensor(gym.acquire_particle_state_tensor(sim))
    gym.refresh_particle_state_tensor(sim)
    biotac_state_init = copy.deepcopy(particle_state_tensor)

    # Set initial indenter state and control target for initial indentation increment
    reset_indenter_state(gym=gym,
                         envs=envs,
                         indenters=indenters,
                         indent_target=indent_target,
                         indent_dist=indent_dist)
    ctrl_target = 0.0
    ctrl_target = set_ctrl_target(gym=gym,
                                  envs=envs,
                                  indenters=indenters,
                                  ctrl_target=ctrl_target,
                                  indent_dist=indent_dist,
                                  indent_steps=indent_steps)

    num_envs = len(envs)
    indent_inc_flags = np.zeros(num_envs)
    indent_dist_flags = np.zeros(num_envs)
    while not gym.query_viewer_has_closed(viewer):

        # Run simulation
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        # Visualize motion and deformation
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)
        gym.clear_lines(viewer)

        # Draw indenter pose and visualize contacts
        for env, indenter in zip(envs, indenters):
            if draw_pose:
                get_pose_and_draw(gym, env, viewer, axes, indenter)
            if show_contacts:
                gym.draw_env_rigid_contacts(viewer, env, gymapi.Vec3(1.0, 0.5, 0.0), 0.05, True)
                gym.draw_env_soft_contacts(viewer, env, gymapi.Vec3(0.6, 0.0, 0.6), 0.05, False, True)

        # Set indentation flags
        env_index = 0
        for env, indenter in zip(envs, indenters):
            indenter_dof_state = gym.get_actor_dof_states(env, indenter, gymapi.STATE_ALL)
            indenter_dof_pos = indenter_dof_state['pos']
            # If indenter joint position exceeds specified indentation increment, set flag
            if abs(ctrl_target - indenter_dof_pos) < err_tol:
                indent_inc_flags[env_index] = 1
            # If indenter joint position exceeds full indentation indentation distance, set flag
            if abs(indent_dist - indenter_dof_pos) < err_tol:
                indent_dist_flags[env_index] = 1
            env_index += 1

        # If all indenter joint positions have exceeded indentation increment targets, 
        # extract results and set new target
        if np.all(indent_inc_flags):
            indent_inc_flags = np.zeros(num_envs)
            results_curr_inc = extract_results(gym=gym,
                                               sim=sim,
                                               envs=envs,
                                               particle_states=particle_state_tensor,
                                               extract_stress=extract_stress)
            results.append(results_curr_inc)
            ctrl_target = set_ctrl_target(gym=gym,
                                          envs=envs,
                                          indenters=indenters,
                                          ctrl_target=ctrl_target,
                                          indent_dist=indent_dist,
                                          indent_steps=indent_steps)

        # If all indenter joint positions have exceeded full indentation distance, 
        # reset BioTac and indenter states and target
        if np.all(indent_dist_flags):
            indent_dist_flags = np.zeros(num_envs)
            reset_biotac_state(gym=gym,
                               sim=sim,
                               biotac_state=biotac_state_init)
            reset_indenter_state(gym=gym,
                                 envs=envs,
                                 indenters=indenters,
                                 indent_target=indent_target,
                                 indent_dist=indent_dist)
            ctrl_target = 0.0
            ctrl_target = set_ctrl_target(gym=gym,
                                          envs=envs,
                                          indenters=indenters,
                                          ctrl_target=ctrl_target,
                                          indent_dist=indent_dist,
                                          indent_steps=indent_steps)
            break

    # Simulate and visualize one final step
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    gym.sync_frame_time(sim)
    gym.clear_lines(viewer)

    # Clean up
    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)

    return results

def set_asset_options():
    """Set asset options common to all assets."""

    options = gymapi.AssetOptions()
    options.flip_visual_attachments = False
    options.armature = 0.0
    options.thickness = 0.0
    options.linear_damping = 0.0
    options.angular_damping = 0.0
    options.default_dof_drive_mode = gymapi.DOF_MODE_POS
    options.min_particle_mass = 1e-20

    return options

def set_biotac_matl_props(base_dir, elast_mod, poiss_ratio):
    """Set the BioTac material properties by copying and modifying a URDF template."""
    # TODO: There may be now be a built-in function for setting material properties in the Gym Python API.

    template_path = os.path.join(base_dir, 'biotac_template.urdf')
    file_path = os.path.join(base_dir, 'biotac.urdf')
    copyfile(template_path, file_path)
    with fileinput.FileInput(file_path, inplace=True) as file_obj:
        for line in file_obj:
            if 'youngs' in line:
                print(line.replace('youngs value=""', f'youngs value="{elast_mod}"'), end='')
            elif 'poissons' in line:
                print(line.replace('poissons value=""', f'poissons value="{poiss_ratio}"'), end='')
            else:
                print(line, end='')

def set_ctrl_props(gym, envs, indenters, pd_gains=[1.0e9, 0.0]):
    """Set the properties for the indenter PD controllers."""

    for env, indenter in zip(envs, indenters):
        indenter_dof_props = gym.get_actor_dof_properties(env, indenter)
        indenter_dof_props['driveMode'][0] = gymapi.DOF_MODE_POS
        indenter_dof_props['stiffness'][0] = pd_gains[0]
        indenter_dof_props['damping'][0] = pd_gains[1]
        gym.set_actor_dof_properties(env, indenter, indenter_dof_props)

def set_ctrl_target(gym, envs, indenters, ctrl_target, indent_dist, indent_steps):
    """Set the controller targets for the next indentation increment."""

    ctrl_target += indent_dist / indent_steps
    for env, indenter in zip(envs, indenters):
        gym.set_actor_dof_position_targets(env, indenter, np.array([ctrl_target], dtype=np.float32))

    return ctrl_target

def set_scene_props(num_envs, env_dim=0.05):
    """Set the scene and environment properties."""

    envs_per_row = int(np.ceil(np.sqrt(num_envs)))
    env_lower = gymapi.Vec3(-env_dim, 0, -env_dim)
    env_upper = gymapi.Vec3(env_dim, env_dim, env_dim)
    scene_props = {'num_envs': num_envs,
                   'per_row': envs_per_row,
                   'lower': env_lower,
                   'upper': env_upper}

    return scene_props

if __name__ == "__main__":
    main()