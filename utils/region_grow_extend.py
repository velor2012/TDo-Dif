'''
                       ::
                      :;J7, :,                        ::;7:
                      ,ivYi, ,                       ;LLLFS:
                      :iv7Yi                       :7ri;j5PL
                     ,:ivYLvr                    ,ivrrirrY2X,
                     :;r@Wwz.7r:                :ivu@kexianli.
                    :iL7::,:::iiirii:ii;::::,,irvF7rvvLujL7ur
                   ri::,:,::i:iiiiiii:i:irrv177JX7rYXqZEkvv17
                ;i:, , ::::iirrririi:i:::iiir2XXvii;L8OGJr71i
              :,, ,,:   ,::ir@mingyi.irii:i:::j1jri7ZBOS7ivv,
                 ,::,    ::rv77iiiriii:iii:i::,rvLq@huhao.Li
             ,,      ,, ,:ir7ir::,:::i;ir:::i:i::rSGGYri712:
           :::  ,v7r:: ::rrv77:, ,, ,:i7rrii:::::, ir7ri7Lri
          ,     2OBBOi,iiir;r::        ,irriiii::,, ,iv7Luur:
        ,,     i78MBBi,:,:::,:,  :7FSL: ,iriii:::i::,,:rLqXv::
        :      iuMMP: :,:::,:ii;2GY7OBB0viiii:i:iii:i:::iJqL;::
       ,     ::::i   ,,,,, ::LuBBu BBBBBErii:i:i:i:i:i:i:r77ii
      ,       :       , ,,:::rruBZ1MBBqi, :,,,:::,::::::iiriri:
     ,               ,,,,::::i:  @arqiao.       ,:,, ,:::ii;i7:
    :,       rjujLYLi   ,,:::::,:::::::::,,   ,:i,:,,,,,::i:iii
    ::      BBBBBBBBB0,    ,,::: , ,:::::: ,      ,,,, ,,:::::::
    i,  ,  ,8BMMBBBBBBi     ,,:,,     ,,, , ,   , , , :,::ii::i::
    :      iZMOMOMBBM2::::::::::,,,,     ,,,,,,:,,,::::i:irr:i:::,
    i   ,,:;u0MBMOG1L:::i::::::  ,,,::,   ,,, ::::::i:i:iirii:i:i:
    :    ,iuUuuXUkFu7i:iii:i:::, :,:,: ::::::::i:i:::::iirr7iiri::
    :     :rk@Yizero.i:::::, ,:ii:::::::i:::::i::,::::iirrriiiri::,
     :      5BMBBBBBBSr:,::rv2kuii:::iii::,:i:,, , ,,:,:i@petermu.,
          , :r50EZ8MBBBBGOBBBZP7::::i::,:::::,: :,:,::i;rrririiii::
              :jujYY7LS0ujJL7r::,::i::,::::::::::::::iirirrrrrrr:ii:
           ,:  :@kevensun.:,:,,,::::i:i:::::,,::::::iir;ii;7v77;ii;i,
           ,,,     ,,:,::::::i:iiiii:i::::,, ::::iiiir@xingjief.r;7:i,
        , , ,,,:,,::::::::iiiiiiiiii:,:,:::::::::iiir;ri7vL77rrirri::
         :,, , ::::::::i:::i:::i:i::,,,,,:,::i:i:::iir;@Secbone.ii:::

author: Wenyi Chen 
postgraduate of Computer Vision,
Wuhan University, Hubei , China 

email: wenyichen@whu.edu.com or wenyichen550@gmail.com
Date: 2021-06-03 10:18:04
Description: file content
'''
import numpy as np
from scipy import interpolate, ndimage, stats
from skimage import morphology
from sklearn import cluster, mixture
import logging
from sklearn.metrics import pairwise_distances
from skimage import measure
def superpixel_centers(segments):
    """ estimate centers of each superpixel
    :param ndarray segments: segmentation np.array<height, width>
    :return [(float, float)]:
    >>> segm = np.array([[0] * 6 + [1] * 5, [0] * 6 + [2] * 5])
    >>> superpixel_centers(segm)
    [(0.5, 2.5), (0.0, 8.0), (1.0, 8.0)]
    >>> superpixel_centers(np.array([segm, segm, segm]))
    [[1.0, 0.5, 2.5], [1.0, 0.0, 8.0], [1.0, 1.0, 8.0]]
    """
    logging.debug('compute centers for %d superpixels', segments.max())
    centers = [list() for _ in range(np.max(segments) + 1)]

    if segments.ndim <= 2:
        # regionprops works for labels from 1
        regions = measure.regionprops(segments + 1)
        for region in regions:
            centers[region['label'] - 1] = region['centroid']
    elif segments.ndim == 3:
        # http://peekaboo-vision.blogspot.cz/2011/08/region-connectivity-graphs-in-python.html
        grids = np.mgrid[:segments.shape[0], :segments.shape[1], :segments.shape[2]]
        # for v in range(len(centers)):
        #     centers[v] = [grids[g][segments == v].mean() for g in range(3)]
        segm_flat = segments.ravel()
        grids_flat = [g.ravel() for g in grids]
        for i, lb in enumerate(segm_flat):
            vals = [grids_flat[g][i] for g in range(3)]
            centers[lb].append(vals)
        for lb, vals in enumerate(centers):
            centers[lb] = np.mean(vals, axis=0).tolist()
    else:
        logging.error('not supported image dim: %r', segments.shape)
    return centers

def make_graph_segm_connect_grid2d_conn4(grid):
    """ construct graph of connected components
    :param ndarray grid: segmentation
    :return [int], [(int, int)]:
    >>> grid = np.array([[0] * 5 + [1] * 5, [2] * 5 + [3] * 5])
    >>> v, edges = make_graph_segm_connect_grid2d_conn4(grid)
    >>> v
    array([0, 1, 2, 3])
    >>> edges
    [[0, 1], [0, 2], [1, 3], [2, 3]]
    """
    # get unique labels
    logging.debug('make graph segment connect edges - 2d conn4')
    vertices = np.unique(grid)
    # map unique labels to [1,...,num_labels]
    reverse_dict = dict(zip(vertices, np.arange(len(vertices))))
    grid = np.array([reverse_dict[x] for x in grid.flat]).reshape(grid.shape)
    all_edges = get_segment_diffs_2d_conn4(grid)
    return make_graph_segment_connect_edges(vertices, all_edges)

def make_graph_segment_connect_edges(vertices, all_edges):
    """ make graph of connencted components
    SEE: http://peekaboo-vision.blogspot.cz/2011/08/region-connectivity-graphs-in-python.html
    :param ndarray vertices:
    :param ndarray all_edges:
    :return tuple(ndarray,ndarray):
    """
    all_edges = all_edges[all_edges[:, 0] != all_edges[:, 1], :]
    all_edges = np.sort(all_edges, axis=1)
    nb_vertices = len(vertices)
    edge_hash = all_edges[:, 0] + nb_vertices * all_edges[:, 1]
    # find unique connections
    edges = np.unique(edge_hash)
    # undo hashing
    edges = [[vertices[int(edge % nb_vertices)], vertices[int(edge / nb_vertices)]] for edge in edges]
    return vertices, edges

def get_segment_diffs_2d_conn4(grid):
    """ wrapper for getting 4-connected in 2D image plane
    :param ndarray grid: segmentation
    :return [(int, int)]:
    """
    down = np.c_[grid[:-1, :].ravel(), grid[1:, :].ravel()]
    right = np.c_[grid[:, :-1].ravel(), grid[:, 1:].ravel()]
    return np.vstack([right, down])

def get_neighboring_segments(edges):
    """ get the indexes of neighboring superpixels for each superpixel
    the input is list edges of all neighboring segments
    :param [[int, int]] edges:
    :return [[int]]:
    >>> get_neighboring_segments([[0, 1], [1, 2], [1, 3], [2, 3]])
    [[1], [0, 2, 3], [1, 3], [1, 2]]
    """
    list_neighbours = np.zeros((np.max(edges) + 1, 0)).tolist()
    for e1, e2 in edges:
        list_neighbours[e1].append(e2)
        list_neighbours[e2].append(e1)
    return list_neighbours

def compute_data_costs_points(slic, slic_prob_fg, centres, labels):
    """ compute Look up Table ro date term costs
    :param nadarray slic: superpixel segmentation
    :param list(float) slic_prob_fg: weight for particular pixel belongs to FG
    :param [[int, int]] centres: actual centre position
    :param list(int) labels: labels for points to be assigned to an object
    :return:
    """
    data_proba = np.empty((len(labels), len(centres) + 1))
    data_proba[:, 0] = 1. - slic_prob_fg
    for i, centre in enumerate(centres):
        data_proba[:, i + 1] = slic_prob_fg
        vertex = slic[centre[0], centre[1]]
        labels[vertex] = i + 1
    # use an offset to avoid 0 in logarithm
    lut_data_cost = -np.log(data_proba + 1e-9)
    lut_data_cost[np.isinf(lut_data_cost)] = GC_REPLACE_INF
    return lut_data_cost, labels
def compute_pairwise_penalty(edges, labels, prob_bg_fg=0.05, prob_fg1_fg2=0.01):
    """ compute cost of neighboring labels pionts
    :param [(int, int)] edges: graph edges, connectivity
    :param [int] labels: labels for vertexes
    :param float prob_bg_fg: penalty between background and foreground
    :param float prob_fg1_fg2: penaly between two different foreground classes
    :return:
    >>> edges = np.array([[0, 1], [1, 2], [0, 3], [2, 3], [2, 4]])
    >>> labels = np.array([0, 0, 1, 2, 1])
    >>> compute_pairwise_penalty(edges, labels, 0.05, 0.01)
    array([ 0.        ,  2.99573227,  2.99573227,  4.60517019,  0.        ])
    """
    edges_labeled = labels[edges]
    is_diff = (edges_labeled[:, 0] != edges_labeled[:, 1])
    is_bg = np.logical_or(edges_labeled[:, 0] == 0, edges_labeled[:, 1] == 0)
    is_bg = np.logical_and(is_diff, is_bg)
    costs = -np.log(prob_fg1_fg2) * is_diff
    costs[is_bg] = -np.log(prob_bg_fg)
    return costs

def enforce_center_labels(slic, labels, centres):
    """ force the labels to hold label of the center,
    prevention of desepearing labels of any center in list
    :param slic:
    :param labels:
    :param centres:
    :return:
    """
    for i, center in enumerate(centres):
        idx = slic[int(center[0]), int(center[1])]
        labels[idx] = i + 1
    return labels

def get_neighboring_candidates(slic_neighbours, labels, object_idx, use_other_obj=True):
    """ get neighboring candidates from background
    and optionally also from foreground if it is allowed
    :param [[int]] slic_neighbours: list of neighboring superpixel for each one
    :param [int] labels: labels for each superpixel
    :param int object_idx:
    :param bool use_other_obj: allowing use another foreground object
    :return [int]:
    >>> neighbours = [[1], [0, 2, 3], [1, 3], [1, 2]]
    >>> labels = np.array([0, 0, 1, 1])
    >>> get_neighboring_candidates(neighbours, labels, 1)
    [1]
    """
    neighbours = []
    for l_idx in np.array(slic_neighbours)[labels == object_idx]:
        neighbours += l_idx
    neighbours = np.unique(neighbours)
    if use_other_obj:
        neighbours = [lb for lb in neighbours if labels[lb] != object_idx]
    else:
        neighbours = [lb for lb in neighbours if labels[lb] == 0]
    return neighbours
    
def compute_rg_crit(
    labels,
    lut_data_cost,
    lut_shape_cost,
    slic_weights,
    edges,
    coef_data,
    coef_shape,
    coef_pairwise,
    prob_label_trans,
):
    all_range = np.arange(len(labels))
    crit_data = coef_data * lut_data_cost[all_range, labels]
    crit_shape = coef_shape * lut_shape_cost[all_range, labels]
    crit = np.sum(slic_weights * (crit_data + crit_shape))
    if coef_pairwise > 0:
        pairwise_costs = compute_pairwise_penalty(edges, labels, prob_label_trans[0], prob_label_trans[1])
        pairwise_costs[np.isinf(pairwise_costs)] = GC_REPLACE_INF
        crit += coef_pairwise * np.sum(pairwise_costs)
    return crit

RG2SP_THRESHOLDS = {
    'centre': 30,  # min center displacement since last iteration
    'shift': 15,  # min rotation change since last iteration
    'volume': 0.1,  # min volume change since last iteration
    'centre_init': 50,  # maximal move from original estimate
}
#: all infinty values in Grah-Cut terms replace by this value
GC_REPLACE_INF = 1e5
def region_growing_slic_greedy(
    slic,
    slic_prob_fg,
    centres,
    shape_model,
    shape_type='cdf',
    coef_data=1.,
    coef_shape=1,
    coef_pairwise=1,
    prob_label_trans=(.1, .01),
    allow_obj_swap=True,
    greedy_tol=1e-3,
    dict_thresholds=None,
    nb_iter=999,
    debug_history=None,
):
    assert len(slic_prob_fg) >= np.max(slic), 'dims of probs %s and slic %s not match' \
                                              % (len(slic_prob_fg), np.max(slic))
    thresholds = RG2SP_THRESHOLDS if dict_thresholds is None else dict_thresholds
    slic_points = superpixel_centers(slic)
    slic_points = np.round(slic_points).astype(int)
    slic_weights = np.bincount(slic.ravel())
    init_centres = np.round(centres).astype(int)

    _, edges = make_graph_segm_connect_grid2d_conn4(slic)
    slic_neighbours = get_neighboring_segments(edges)
    labels = np.zeros(len(slic_points), dtype=int)

    return labels
    
def regionGrow(img,thresh=None,p = 1):
    height, weight,c = img.shape
    segments = slic(img, n_segments = 1000, sigma = 1)
    slic_points = superpixel_centers(segments)
    uni_labels = np.unique(segments)
    seedMark = np.zeros(len(uni_labels),dtype=np.int64) - 1
    feats = np.array([np.mean(img[segments == l,:],axis=0) for l in uni_labels])
    dist = pairwise_distances(feats, feats,metric='euclidean') #dist为NXK
    sim = np.max(dist) - dist
    init_seeds = [700]
    if thresh is None:
        thresh = -np.sort(-sim,axis=1)[:,:20]
        thresh = np.sort(thresh.flatten())[0]
    #connects = selectConnects(p)
    _, edges = make_graph_segm_connect_grid2d_conn4(segments)
    slic_neighbours = get_neighboring_segments(edges)
    while len(init_seeds)>0:
        init_seed = init_seeds.pop(0)
        seedList = []
        seedList.append(init_seed)
        label = uni_labels[init_seed]
        while(len(seedList)>0):
            currentPoint = seedList.pop(0)#弹出第一个元素
            seedMark[currentPoint] = label
            for i in slic_neighbours[currentPoint]:
                if seedMark[i] == -1 and sim[currentPoint,i] > thresh and sim[init_seed,i] > thresh:
                    # print('sim:{}'.format(sim[currentPoint,i]))
                    seedMark[i] = int(label)
                    seedList.append(i)

    segments_cp = segments.copy()
    for i in range(len(uni_labels)):
        if seedMark[i] != -1 :
            segments[segments == i] = seedMark[i]
    return segments_cp,segments

if __name__ == "__main__":
    import argparse
    from skimage.segmentation import slic
    from skimage.util import img_as_float
    from skimage.segmentation import mark_boundaries
    from skimage import io
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", default="/mnt/data/cwy/DeepLabV3Plus-Pytorch-master/1508039851.0_start_13m32s_frame_000108.png", help = "Path to the image")
    args = vars(ap.parse_args())
    image = img_as_float(io.imread(args["image"]))
    # segments = slic(image, n_segments = 100, sigma = 1)
    # label = region_growing_slic_greedy(segments,np.ones(len(np.unique(segments)))-0.1,[(76,40),(370,50),(200,200)],None)
    labels_o,labels = regionGrow(image)


    io.imsave('test2.png', mark_boundaries(image, labels_o))
    io.imsave('test.png', mark_boundaries(image, labels))
