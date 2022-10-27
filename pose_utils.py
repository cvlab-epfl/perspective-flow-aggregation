import numpy as np

def compute_pose_diff(mesh3ds, K, gtR, gtT, predR, predT, isSym=False):
    ptCnt = len(mesh3ds)
    if ptCnt > 1000:
        tmp_index = np.random.choice(len(mesh3ds), 1000, replace=True)
        mesh3ds = mesh3ds[tmp_index]
        ptCnt = 1000
    #
    pred_3d1 = (np.matmul(gtR, mesh3ds.T) + gtT).T
    pred_3d2 = (np.matmul(predR, mesh3ds.T) + predT).T

    # find the closest point for symmetric objects
    if isSym:
        ext_3d1 = pred_3d1.repeat(ptCnt, axis=0)
        ext_3d2 = pred_3d2.reshape(1, -1).repeat(ptCnt, axis=0).reshape(-1, 3)
        min_idx2 = np.argmin(np.linalg.norm(ext_3d1-ext_3d2, axis=1).reshape(ptCnt, -1), axis=1)
        pred_3d2 = ext_3d2[min_idx2]

    p = np.matmul(K, pred_3d1.T)
    p[0] = p[0] / (p[2] + 1e-8)
    p[1] = p[1] / (p[2] + 1e-8)
    pred_2d1 = p[:2].T

    p = np.matmul(K, pred_3d2.T)
    p[0] = p[0] / (p[2] + 1e-8)
    p[1] = p[1] / (p[2] + 1e-8)
    pred_2d2 = p[:2].T

    error_3d = np.linalg.norm(pred_3d1 - pred_3d2, axis=1).mean()
    error_2d = np.linalg.norm(pred_2d1 - pred_2d2, axis=1).mean()

    return error_3d, error_2d

def evalute_auc_metric(error_3ds, max_err):
    error_3ds = np.array(error_3ds)
    sampleCnt = len(error_3ds)
    if sampleCnt == 0:
        return 0
    binCnt = 1000
    total_auc = 0.0
    for i in range(binCnt):
        validCnt = (error_3ds <= ((i+1) * (max_err/binCnt))).sum()
        binContrib = ((validCnt / sampleCnt) * (1 / binCnt))
        total_auc += binContrib
    return total_auc

def evaluate_pose_predictions(predictions, class_number, meshes, mesh_diameters, symmetry_types={}):
    INF = 100000000
    classNum = class_number - 1 # get rid of the background class

    thresholds_adi = [0.05, 0.10, 0.20, 0.50]
    thresholds_rep = [2, 5, 10, 20]
        
    accuracy_adi_per_class = []
    accuracy_auc_per_class = []
    accuracy_rep_per_class = []
    # 
    depth_bins = 3
    accuracy_adi_per_depth = []
    accuracy_rep_per_depth = []

    surfacePts = []
    for ms in meshes:
        pts = np.array(ms[0].vertices)
        tmp_index = np.random.choice(len(pts), 1000, replace=True)
        pts = pts[tmp_index]
        surfacePts.append(pts)

    # get depth range from annotations, and divide it to serval bins
    depth_min = INF
    depth_max = 0
    for filename, item in predictions.items():
        gtTs = np.array(item['meta']['translations'])
        for T in gtTs:
            depth = float(T.reshape(-1)[2])
            depth_min = min(depth_min, depth)
            depth_max = max(depth_max, depth)
    depth_max += 1e-5 # add some margin for safe depth index computation
    depth_bin_width = (depth_max - depth_min) / depth_bins

    errors_adi_per_depth = list([] for i in range(0, depth_bins))
    errors_rep_per_depth = list([] for i in range(0, depth_bins))
    for clsid in range(classNum):
        isSym = (("cls_" + str(clsid)) in symmetry_types)
        errors_adi_all = [] # 3D errors, %
        errors_abs3d_all = [] # 3D erros, absolute
        errors_rep_all = [] # 2D errors
        depth_all = [] # depth for each sample
        object_cx_all = []
        object_cy_all = []
        # 
        for filename, item in predictions.items():
            K = np.array(item['meta']['K'])
            pred = item['pred']
            gtIDs = item['meta']['class_ids']
            gtRs = np.array(item['meta']['rotations'])
            gtTs = np.array(item['meta']['translations'])
            
            # filter by class id
            pred = [p for p in pred if p[1] == clsid]
            gtIdx = [gi for gi in range(len(gtIDs)) if gtIDs[gi] == clsid]
            if len(gtIdx) == 0:
                continue

            # find predictions with best confidences
            assert(len(gtIdx) == 1) # only one object for one class now

            # get the depth bin of the object
            gi = gtIdx[0] # only pick up the first one
            depth = float(gtTs[gi].reshape(-1)[2])
            depth_idx = int((depth - depth_min) / depth_bin_width)
            depth_all.append(depth)
            # 
            if len(pred) > 0:
                # find the best confident one
                bestIdx = 0
                R1 = gtRs[gi]
                T1 = gtTs[gi]
                R2 = np.array(pred[bestIdx][2])
                T2 = np.array(pred[bestIdx][3])
                err_3d, err_2d = compute_pose_diff(surfacePts[clsid], K, R1, T1, R2, T2, isSym=isSym)
                #
                # get the reprojected center
                tmp_pt = np.matmul(K, T1)
                object_cx = tmp_pt[0] / tmp_pt[2]
                object_cy = tmp_pt[1] / tmp_pt[2]
                object_cx_all.append(float(object_cx))
                object_cy_all.append(float(object_cy))
                # 
                errors_adi_all.append(err_3d / mesh_diameters[clsid])
                errors_abs3d_all.append(err_3d)
                errors_rep_all.append(err_2d)
                errors_adi_per_depth[depth_idx].append(err_3d / mesh_diameters[clsid])
                errors_rep_per_depth[depth_idx].append(err_2d)
            else:
                object_cx_all.append(-1)
                object_cy_all.append(-1)
                errors_adi_all.append(1.0)
                errors_abs3d_all.append(1e10)
                errors_rep_all.append(50)
                errors_adi_per_depth[depth_idx].append(1.0)
                errors_rep_per_depth[depth_idx].append(50)
        # 
        auc = evalute_auc_metric(errors_abs3d_all, max_err=100)
        #
        assert(len(errors_adi_all) == len(errors_rep_all))
        counts_all = len(errors_adi_all)
        if counts_all > 0:
            accuracy = {}
            for th in thresholds_adi:
                validCnt = (np.array(errors_adi_all) < th).sum()
                key = 'ADI' + ("%.2fd" % th).lstrip('0')
                accuracy[key] = (validCnt / counts_all) * 100
            accuracy_adi_per_class.append(accuracy)
            # 
            accuracy = {}
            accuracy['AUC    '] = auc * 100
            accuracy_auc_per_class.append(accuracy)
            # 
            accuracy = {}
            for th in thresholds_rep:
                validCnt = (np.array(errors_rep_all) < th).sum()
                accuracy[('REP%02dpx'%th)] = (validCnt / counts_all) * 100
            accuracy_rep_per_class.append(accuracy)
        else:
            accuracy_adi_per_class.append({})
            accuracy_auc_per_class.append({})
            accuracy_rep_per_class.append({})
    # 
    # compute accuracy for every depth bin
    for i in range(depth_bins):
        assert(len(errors_adi_per_depth[i]) == len(errors_rep_per_depth[i]))
        counts_all = len(errors_adi_per_depth[i])
        if counts_all > 0:
            accuracy = {}
            for th in thresholds_adi:
                validCnt = (np.array(errors_adi_per_depth[i]) < th).sum()
                key = 'ADI' + ("%.2fd" % th).lstrip('0')
                accuracy[key] = (validCnt / counts_all) * 100
            accuracy_adi_per_depth.append(accuracy)
            accuracy = {}
            for th in thresholds_rep:
                validCnt = (np.array(errors_rep_per_depth[i]) < th).sum()
                accuracy[('REP%02dpx'%th)] = (validCnt / counts_all) * 100
            accuracy_rep_per_depth.append(accuracy)
        else:
            accuracy_adi_per_depth.append({})
            accuracy_rep_per_depth.append({})
    # 
    return accuracy_adi_per_class, accuracy_auc_per_class, accuracy_rep_per_class, accuracy_adi_per_depth, accuracy_rep_per_depth, [depth_min, depth_max]

def print_accuracy_per_class(accuracy_adi_per_class,  accuracy_auc_per_class, accuracy_rep_per_class):
    assert(len(accuracy_adi_per_class) == len(accuracy_rep_per_class))
    classNum = len(accuracy_adi_per_class)

    firstMeet = True

    for clsIdx in range(classNum):
        if len(accuracy_adi_per_class[clsIdx]) == 0:
            continue

        if firstMeet:
            adi_keys = accuracy_adi_per_class[clsIdx].keys()
            auc_keys = accuracy_auc_per_class[clsIdx].keys()
            rep_keys = accuracy_rep_per_class[clsIdx].keys()

            titleLine = "\t"
            for k in adi_keys:
                titleLine += (k + ' ')
            for k in auc_keys:
                titleLine += (k + ' ')
            for k in rep_keys:
                titleLine += (k + ' ')
            print(titleLine)

            firstMeet = False

        line_per_class = ("cls_%02d" % clsIdx)
        for k in adi_keys:
            line_per_class += ('\t%.2f' % accuracy_adi_per_class[clsIdx][k])
        for k in auc_keys:
            line_per_class += ('\t%.2f' % accuracy_auc_per_class[clsIdx][k])
        for k in rep_keys:
            line_per_class += ('\t%.2f' % accuracy_rep_per_class[clsIdx][k])
        print(line_per_class)
