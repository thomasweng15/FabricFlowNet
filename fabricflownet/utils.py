import cv2
from softgym.envs.bimanual_env import uv_to_world_pos
import numpy as np
from copy import deepcopy
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib as mpl

def remove_occluded_knots(camera_params, knots, coords, depth, zthresh=0.001, debug_viz=False):
    if depth.shape[0] < camera_params['default_camera']['height']:
        print('Warning: resizing depth')
        depth = cv2.resize(depth, (camera_params['default_camera']['height'], camera_params['default_camera']['width']))

    unoccluded_knots = []
    occluded_knots = []
    for i, uv in enumerate(knots):
        u_float, v_float = uv[0], uv[1]
        if np.isnan(u_float) or np.isnan(v_float):
            continue
        u, v = int(np.rint(u_float)), int(np.rint(v_float))

        # check if pixel is outside of image bounds
        if u < 0 or v < 0 or u >= depth.shape[1] or v >= depth.shape[0]:
            knots[i] = [float('NaN'), float('NaN')]
            continue

        # Get depth into world coordinates
        d = depth[v, u]
        deproj_coords = uv_to_world_pos(camera_params, depth, u_float, v_float, particle_radius=0, on_table=False)[0:3]
        zdiff = deproj_coords[1] - coords[i][1]

        # Check is well projected xyz point
        if zdiff > zthresh:
            # invalidate u, v and continue
            occluded_knots.append(deepcopy(knots[i]))
            knots[i] = [float('NaN'), float('NaN')]
            continue

        unoccluded_knots.append(deepcopy(knots[i]))
    
    # Debug plotting
    if debug_viz:
        # 3D scatterplot
        # fig = plt.figure()
        # ax = fig.add_subplot(1, 1, 1, projection='3d')
        # for i, (u, v) in enumerate(knots[::3]):
        #     c = 'r' if np.isnan(u) or np.isnan(v) else 'b'
        #     ax.scatter(coords[i, 0], coords[i, 2], coords[i, 1], s=1, c=c)
        # plt.show()

        # 2D plot
        fig, ax = plt.subplots(1, 3, figsize=(8, 3))
        ax[0].set_title('depth')
        ax[0].imshow(depth)
        ax[1].set_title('occluded points\nin red')
        ax[1].imshow(depth)
        if occluded_knots != []:
            occluded_knots = np.array(occluded_knots)
            ax[1].scatter(occluded_knots[:, 0], occluded_knots[:, 1], marker='.', s=1, c='r', alpha=0.4)
        ax[2].imshow(depth)
        ax[2].set_title('unoccluded points\nin blue')
        unoccluded_knots = np.array(unoccluded_knots)
        ax[2].scatter(unoccluded_knots[:, 0], unoccluded_knots[:, 1], marker='.', s=1, alpha=0.4)
        plt.show()
        
    return knots

def get_harris(mask, thresh=0.2):
    """Harris corner detector
    Params
    ------
        - mask: np.float32 image of 0.0 and 1.0
        - thresh: threshold for filtering small harris values    Returns
    -------
        - harris: np.float32 array of
    """
    # Params for cornerHarris: 
    # mask - Input image, it should be grayscale and float32 type.
    # blockSize - It is the size of neighbourhood considered for corner detection
    # ksize - Aperture parameter of Sobel derivative used.
    # k - Harris detector free parameter in the equation.
    # https://docs.opencv.org/master/dd/d1a/group__imgproc__feature.html#gac1fc3598018010880e370e2f709b4345
    harris = cv2.cornerHarris(mask, blockSize=5, ksize=5, k=0.01)
    harris[harris<thresh*harris.max()] = 0.0 # filter small values
    harris[harris!=0] = 1.0
    harris_dilated = cv2.dilate(harris, kernel=np.ones((7,7),np.uint8))
    harris_dilated[mask == 0] = 0
    return harris_dilated

def plot_flow(ax, flow_im, skip=0.25):
    """Plot flow as a set of arrows on an existing axis.
    """
    h,w,c = flow_im.shape
    bg = np.zeros((h, w, 3))
    ax.imshow(bg)
    ys, xs, _ = np.where(flow_im != 0)
    n = len(xs)

    inds = np.random.choice(np.arange(n), size=int(n*skip), replace=False)
    flu = flow_im[ys[inds], xs[inds], 1]
    flv = flow_im[ys[inds], xs[inds], 0]
    mags = np.linalg.norm(flow_im[ys[inds], xs[inds], :], axis=1)
    norm = mpl.colors.Normalize()
    norm.autoscale(mags)
    cm = mpl.cm.autumn
    
    ax.quiver(xs[inds], ys[inds], flu, flv, 
                alpha=0.9, color=cm(norm(mags)), angles='xy', 
                scale_units='xy', scale=1,
                headwidth=10, headlength=10, width=0.0025)

def flow_affinewarp(flow_im, angle, dx, dy):
    """Affine transformation of flow image and per-pixel flow vectors.    

    Parameters
    -----------
    flow_im : np.ndarray
        Flow image
        
    angle : float
        Angle in degrees of rotation    
        
    dx : int
        delta x translation    
        
    dy : int
        delta y translation    

    Returns
    -------
    flow_pxrot: np.ndarray
        Flow image transformed both on image level and per-pixel level.
    """
    # Translate and rotate image
    h, w, _ = flow_im.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    M[:2, 2] += [dx, dy]
    flow_im_tf = cv2.warpAffine(flow_im, M, (w, h), flags=cv2.INTER_NEAREST)

    # Rotate per-pixel flow values
    R = M[:2, :2].T
    px = np.reshape(flow_im_tf, (h*w, 2)).T # (2 x h*w) pixel values
    flow_px_tf = np.reshape((R @ px).T, (h, w, 2))
    return flow_px_tf

def unravel_index(indices, shape):
    r"""Converts flat indices into unraveled coordinates in a target shape.

    This is a `torch` implementation of `numpy.unravel_index`.

    Args:
        indices: A tensor of (flat) indices, (*, N).
        shape: The targeted shape, (D,).

    Returns:
        The unraveled coordinates, (*, N, D).
    """

    coord = []

    for dim in reversed(shape):
        coord.append(indices % dim)
        indices = indices // dim

    coord = torch.stack(coord[::-1], dim=-1)

    return coord

def get_flow_place_pt(u,v, flow):
    flow_u_idxs = np.argwhere(flow[0,:,:])
    flow_v_idxs = np.argwhere(flow[1,:,:])
    nearest_u_idx = flow_u_idxs[((flow_u_idxs - [u,v])**2).sum(1).argmin()]
    nearest_v_idx = flow_v_idxs[((flow_v_idxs - [u,v])**2).sum(1).argmin()]

    flow_u = flow[0,nearest_u_idx[0],nearest_u_idx[1]]
    flow_v = flow[1,nearest_v_idx[0],nearest_v_idx[1]]

    new_u = u + flow_u
    new_v = v + flow_v

    return new_u,new_v

def get_gaussian(u, v, sigma=5, size=200):
    x0, y0 = u, v
    num = torch.arange(size).float()
    x, y = num, num
    gx = torch.exp(-(x-x0)**2/(2*sigma**2))
    gy = torch.exp(-(y-y0)**2/(2*sigma**2))
    g = torch.outer(gx, gy)
    g = (g - g.min())/(g.max() - g.min())
    g = g.unsqueeze(0)

    return g

def action_viz(img, action, unmasked_pred):
    ''' img: cv2 image
        action: pick1, place1, pick2, place2
        unmasked_pred: pick1_pred, pick2_pred'''
    pick1, place1, pick2, place2 = action
    pick1_pred, pick2_pred = unmasked_pred

    # draw the original predictions
    u,v = pick1_pred
    cv2.drawMarker(img, (int(u), int(v)), (0,0,200), markerType=cv2.MARKER_STAR, 
                    markerSize=10, thickness=2, line_type=cv2.LINE_AA)
    u,v = pick2_pred
    cv2.drawMarker(img, (int(u), int(v)), (0,0,200), markerType=cv2.MARKER_STAR, 
                    markerSize=10, thickness=2, line_type=cv2.LINE_AA)

    # draw the masked action
    u1,v1 = pick1
    u2,v2 = place1
    cv2.circle(img, (int(u1),int(v1)), 6, (0,200,0), 2)
    cv2.arrowedLine(img, (int(u1),int(v1)), (int(u2),int(v2)), (0, 200, 0), 2)
    u1,v1 = pick2
    u2,v2 = place2
    cv2.circle(img, (int(u1),int(v1)), 6, (0,200,0), 2)
    cv2.arrowedLine(img, (int(u1),int(v1)), (int(u2),int(v2)), (0, 200, 0), 2)

    return img

class Flow:
    def get_flow_image(self, uv_obs, uv_goal_float):
        # Compute uv diff
        uv_diff = np.rint((uv_goal_float - uv_obs)/719*199)
        non_nan_idxs = ~np.isnan(uv_diff).any(axis=1)
        uv_diff_nonan = uv_diff[non_nan_idxs]
        uv_obs_nonan = np.rint(uv_obs[non_nan_idxs]/719*199).astype(int)

        # Make diff image
        im_diff = np.zeros((200, 200, 2))
        im_diff[uv_obs_nonan[:, 1], uv_obs_nonan[:, 0], :] = uv_diff_nonan[:, [1, 0]]
        return im_diff