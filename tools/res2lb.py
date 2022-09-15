import json

def res2lb(res,ori_lb,score_thr,with_score=False,with_largest = False):
    psd_ann = []
    img_id_list = []
    img2psd = {}
    ann_id  = 0
    for ann in res:
        annotation = dict()
        annotation["image_id"] = ann['image_id']
        annotation["segmentation"] = []
        annotation["bbox"] = [i for i in ann['bbox']]
        annotation["category_id"] = ann['category_id']
        annotation["id"] = ann_id
        annotation["iscrowd"] = 0
        annotation["area"] =annotation["bbox"][2] * annotation["bbox"][3]
        annotation["ignore"] = 0
        if with_score or with_largest:
            annotation['score'] = ann['score']

        if ann['image_id'] not in img2psd.keys():
            img2psd[ann['image_id']] = []
        img2psd[ann['image_id']].append(annotation)

        if ann['score'] > score_thr:
            psd_ann.append(annotation)
            ann_id+=1
            if ann['image_id'] not in img_id_list:
                img_id_list.append(ann['image_id'])
    if with_largest:
        for img_id in img2psd.keys():
            if img_id not in img_id_list:
                ann_list = sorted(img2psd[img_id],key=lambda x:x['score'],reverse=True)
                img_id_list.append(img_id)
                psd_ann.append(ann_list[0])
            
    psd_coco = dict()
    psd_coco['images'] =[img for img in ori_lb['images'] if img['id'] in img_id_list]
    psd_coco['categories'] = ori_lb['categories']
    psd_coco['annotations'] = psd_ann
    return psd_coco



if __name__ == '__main__':
    ori_lb_path = '/home/jwl/code/MS_voc/data/VOCdevkit/anno_coco/voc07_trainval.json'
    res_path = '/home/jwl/code/MS_voc/result.bbox.json'
    psd_lb_path = '/home/jwl/code/MS_voc/data/VOCdevkit/anno_coco/psd_train.json'
    score_thr = 0.3

    res = json.load(open(res_path,'r'))
    ori_lb = json.load(open(ori_lb_path,'r'))

    # psd_ann = []
    # ann_id  = 0
    # for ann in res:
    #     if ann['score'] > score_thr:
    #         annotation = dict()
    #         annotation["image_id"] = ann['image_id']
    #         annotation["segmentation"] = []
    #         annotation["bbox"] = [int(i) for i in ann['bbox']]
    #         annotation["category_id"] = ann['category_id']
    #         annotation["id"] = ann_id
    #         annotation["iscrowd"] = 0
    #         annotation["area"] =annotation["bbox"][2] * annotation["bbox"][3]
    #         annotation["ignore"] = 0
    #         psd_ann.append(annotation)
    #         ann_id+=1
    #
    # new_coco = dict()
    # new_coco['images'] = ori_lb['images']
    # new_coco['type'] = ori_lb['type']
    # new_coco['categories'] = ori_lb['categories']
    # new_coco['annotations'] = psd_ann
    psd_coco = res2lb(res,ori_lb,score_thr)
    json_str = json.dumps(psd_coco)
    with open(psd_lb_path, 'w') as json_file:
        json_file.write(json_str)

    pass
