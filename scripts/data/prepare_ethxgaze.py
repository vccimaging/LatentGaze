import os
import csv
import cv2
import numpy as np
import argparse
from tqdm import tqdm

def normalizeData(img, face_model, hr, ht, gc, cam):
    # Normalized camera parameters
    focal_norm = 12800  #[cite: 1]
    distance_norm = 1000  #[cite: 1]
    roiSize = (576, 576)  #[cite: 1]

    ht = ht.reshape((3, 1)) #[cite: 1]
    gc = gc.reshape((3, 1)) #[cite: 1]
    hR = cv2.Rodrigues(hr)[0] #[cite: 1]
    Fc = np.dot(hR, face_model.T) + ht #[cite: 1]
    re = 0.5 * (Fc[:, 0] + Fc[:, 1]).reshape((3, 1))  # center of right eye[cite: 1]
    le = 0.5 * (Fc[:, 2] + Fc[:, 3]).reshape((3, 1))  # center of left eye[cite: 1]

    data = [] #[cite: 1]
    for i, et in enumerate([re, le]): #[cite: 1]
        distance = np.linalg.norm(et) #[cite: 1]
        z_scale = distance_norm / distance #[cite: 1]
        cam_norm = np.array([
            [focal_norm, 0, roiSize[0] / 2],
            [0, focal_norm, roiSize[1] / 2],
            [0, 0, 1.0],
        ]) #[cite: 1]
        S = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, z_scale]]) #[cite: 1]

        hRx = hR[:, 0] #[cite: 1]
        forward = (et / distance).reshape(3) #[cite: 1]
        down = np.cross(forward, hRx) #[cite: 1]
        down /= np.linalg.norm(down) #[cite: 1]
        right = np.cross(down, forward) #[cite: 1]
        right /= np.linalg.norm(right) #[cite: 1]
        R = np.c_[right, down, forward].T #[cite: 1]

        W = np.dot(np.dot(cam_norm, S), np.dot(R, np.linalg.inv(cam))) #[cite: 1]
        img_warped = cv2.warpPerspective(img, W, roiSize) #[cite: 1]

        gc_normalized = gc - et #[cite: 1]
        gc_normalized = np.dot(R, gc_normalized) #[cite: 1]
        gc_normalized = gc_normalized / np.linalg.norm(gc_normalized) #[cite: 1]
        data.append([img_warped, gc_normalized]) #[cite: 1]
        
    return data

def load_splits(splits_dir):
    valid_dict = {}
    for split_name in ['train.csv', 'test.csv']:
        split_path = os.path.join(splits_dir, split_name)
        if not os.path.exists(split_path):
            print(f"Warning: Split file not found -> {split_path}")
            continue
            
        with open(split_path, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]: # Skip header
                parts = line.strip().replace(',', ' ').split()
                if len(parts) < 5: continue
                subj, frame, cam, valid_r, valid_l = parts[0], parts[1], parts[2], int(parts[3]), int(parts[4])
                
                if subj not in valid_dict:
                    valid_dict[subj] = {}
                valid_dict[subj][(frame, f"{cam}.JPG")] = (valid_r, valid_l)
                
    return valid_dict

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    
    print('Loading camera parameters and face model...')
    calib_dir = os.path.join(args.raw_dir, 'calibration')
    camera_matrix = [] #[cite: 1]
    for cam_id in range(2): #[cite: 1]
        xml_file = os.path.join(calib_dir, 'cam_calibration', f'cam{cam_id:02d}.xml') #[cite: 1]
        fs = cv2.FileStorage(xml_file, cv2.FILE_STORAGE_READ) #[cite: 1]
        camera_matrix.append(fs.getNode('Camera_Matrix').mat()) #[cite: 1]
        fs.release() #[cite: 1]
        
    face_model_load = np.loadtxt(os.path.join(calib_dir, 'face_model.txt')) #[cite: 1]
    landmark_use = [20, 23, 26, 29, 15, 19] #[cite: 1]
    face_model = face_model_load[landmark_use, :] #[cite: 1]
    
    valid_dict = load_splits(args.splits_dir)
    print(f'Loaded target splits for {len(valid_dict)} subjects.')

    for subject in tqdm(valid_dict.keys(), desc="Processing subjects"):
        raw_csv_path = os.path.join(args.raw_dir, 'data', 'annotation_train', f'{subject}.csv') #[cite: 1]
        if not os.path.exists(raw_csv_path):
            # Try searching in test directory just in case
            raw_csv_path = os.path.join(args.raw_dir, 'data', 'annotation_test', f'{subject}.csv')
            if not os.path.exists(raw_csv_path):
                continue
                
        out_csv_path = os.path.join(args.output_dir, f"{subject}.csv")
        
        with open(raw_csv_path, 'r') as in_f, open(out_csv_path, 'w', newline='') as out_f:
            reader = csv.reader(in_f, dialect='excel') #[cite: 1]
            writer = csv.writer(out_f, dialect='excel') #[cite: 1]
            # Write expected header for LMDB creation
            writer.writerow(['frame', 'camera.JPG', 'gr_x', 'gr_y', 'gr_z', 'gl_x', 'gl_y', 'gl_z', 'valid_r', 'valid_l'])
            
            for row in reader:
                frame = row[0] #[cite: 1]
                camim = row[1] #[cite: 1]
                
                if (frame, camim) not in valid_dict[subject]:
                    continue
                    
                valid_r, valid_l = valid_dict[subject][(frame, camim)]
                
                if camim == 'cam00.JPG': #[cite: 1]
                    eye_left = True #[cite: 1]
                elif camim == 'cam01.JPG': #[cite: 1]
                    eye_left = False #[cite: 1]
                else:
                    continue #[cite: 1]
                    
                srcimfile = os.path.join(args.raw_dir, 'data', 'train', subject, frame, camim) #[cite: 1]
                if not os.path.exists(srcimfile):
                    continue
                    
                img = cv2.imread(srcimfile) #[cite: 1]
                
                gc = np.array([float(row[4]), float(row[5]), float(row[6])]).reshape(3, 1) #[cite: 1]
                hr = np.array([float(row[7]), float(row[8]), float(row[9])]).reshape(3, 1) #[cite: 1]
                ht = np.array([float(row[10]), float(row[11]), float(row[12])]).reshape(3, 1) #[cite: 1]
                
                data = normalizeData(img, face_model, hr, ht, gc, camera_matrix[1 - eye_left]) #[cite: 1]
                
                # Retrieve parsed gaze directions
                gr = data[0][1].reshape(3).tolist()
                gl = data[1][1].reshape(3).tolist()
                
                # Save crop images following expected directory structure: data/crops/<subject>/<frame>/<camera>_{r,l}.JPG
                frame_out_dir = os.path.join(args.output_dir, subject, frame)
                os.makedirs(frame_out_dir, exist_ok=True)
                
                if valid_r == 1:
                    tgt_r = os.path.join(frame_out_dir, f"{camim[:-4]}_r.JPG") #[cite: 1]
                    cv2.imwrite(tgt_r, data[0][0]) #[cite: 1]
                if valid_l == 1:
                    tgt_l = os.path.join(frame_out_dir, f"{camim[:-4]}_l.JPG") #[cite: 1]
                    cv2.imwrite(tgt_l, data[1][0]) #[cite: 1]
                    
                # Write to corresponding <subject>.csv
                writer.writerow([frame, camim, gr[0], gr[1], gr[2], gl[0], gl[1], gl[2], valid_r, valid_l])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract ETH-XGaze eye crops based on predefined splits.")
    parser.add_argument('--raw-dir', type=str, required=True, help="Path to original ETH-XGaze root directory.")
    parser.add_argument('--splits-dir', type=str, default='data/splits', help="Directory containing train.csv and test.csv")
    parser.add_argument('--output-dir', type=str, default='data/crops', help="Directory to output the crops and CSVs.")
    args = parser.parse_args()
    
    main(args)
