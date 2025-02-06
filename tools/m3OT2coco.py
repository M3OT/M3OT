import argparse
import os
import os.path as osp
from collections import defaultdict
import mmcv
from tqdm import tqdm

USELESS = [5, 6, 9, 10, 11]
IGNORES = [7, 8, 12, 13]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert MOT gt.txt to COCO format.')
    parser.add_argument('-i', '--input', help='path of MOT data')
    parser.add_argument('-o', '--output', help='path to save COCO formatted label file')
    return parser.parse_args()


def parse_gts(gt_path):
    outputs = defaultdict(list)
    gts = mmcv.list_from_file(gt_path)
    for gt in gts:
        gt = gt.strip().split(',')
        frame_id, ins_id = map(int, gt[:2])
        bbox = list(map(float, gt[2:6]))  # bbox: [x1, y1, w, h]

         
        bbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]

        conf = float(gt[6])
        class_id = int(gt[7])
        visibility = float(gt[8])

        if class_id in USELESS:
            continue
        anns = dict(
            category_id=class_id,
            bbox=bbox,  
            area=bbox[2] * bbox[3],
            iscrowd=False,
            visibility=visibility,
            mot_instance_id=ins_id,
            mot_conf=conf,
            mot_class_id=class_id)
        outputs[frame_id].append(anns)
    return outputs


def find_gt_and_images(root_dir):
    
    video_paths = []
    for root, dirs, files in os.walk(root_dir):
        
        parts = root.split(os.sep)
        if '2' in parts and 'gt' in dirs and 'img1' in dirs:
            gt_path = osp.join(root, 'gt', 'gt.txt')
            img_dir = osp.join(root, 'img1')
            video_paths.append((gt_path, img_dir))
    return video_paths


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    sets = ['train', 'val', 'test']
    vid_id, img_id, ann_id = 1, 1, 1

    for subset in sets:
        print(f'Converting {subset} set to COCO format')
        subset_dir = osp.join(args.input, subset)
        out_file = osp.join(args.output, f'{subset}_cocoformat.json')

        outputs = dict(images=[], annotations=[], categories=[], videos=[])
        outputs['categories'] = [dict(id=1, name='vehicle')]  
        video_paths = find_gt_and_images(subset_dir)

        for gt_path, img_dir in tqdm(video_paths):
            video_name = osp.basename(osp.dirname(gt_path))              img_files = sorted(os.listdir(img_dir))
            img2gts = parse_gts(gt_path)
            width, height = 640, 512  

            video_info = dict(
                id=vid_id, name=video_name, fps=30, width=width, height=height)
            outputs['videos'].append(video_info)

            ins_maps = dict()
            for frame_id, img_file in enumerate(img_files):
                img_path = osp.join(img_dir, img_file)
                mot_frame_id = int(osp.splitext(img_file)[0])
                image_info = dict(
                    id=img_id,
                    video_id=vid_id,
                    file_name=img_path,
                    height=height,
                    width=width,
                    frame_id=frame_id,
                    mot_frame_id=mot_frame_id)

                gts = img2gts.get(mot_frame_id, [])
                for gt in gts:
                    gt.update(id=ann_id, image_id=img_id)
                    mot_ins_id = gt['mot_instance_id']
                    if mot_ins_id in ins_maps:
                        gt['instance_id'] = ins_maps[mot_ins_id]
                    else:
                        ins_maps[mot_ins_id] = ann_id
                        gt['instance_id'] = ann_id
                    outputs['annotations'].append(gt)
                    ann_id += 1

                outputs['images'].append(image_info)
                img_id += 1

            vid_id += 1

        mmcv.dump(outputs, out_file)
        print(f'Done! Saved as {out_file}')


if __name__ == '__main__':
    main()
