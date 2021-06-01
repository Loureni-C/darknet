def compute_AP_COCO(annotation_records, gt_classes_records, pred_classes_records, class_names, class_filter=None, show_result=True):
    '''
    Compute MSCOCO AP list on AP 0.5:0.05:0.95
    '''
    iou_threshold_list = np.arange(0.50, 1.00, 0.05)
    APs = {}
    pbar = tqdm(total=len(iou_threshold_list), desc='Eval COCO')
    for iou_threshold in iou_threshold_list:
        iou_threshold = round(iou_threshold, 2)
        mAP, mAPs = compute_mAP_PascalVOC(annotation_records, gt_classes_records, pred_classes_records, class_names, iou_threshold, show_result=False)

        if class_filter is not None:
            mAP = get_filter_class_mAP(mAPs, class_filter, show_result=False)

        APs[iou_threshold] = round(mAP, 6)
        pbar.update(1)

    pbar.close()

    #sort AP result by value, in descending order
    APs = OrderedDict(sorted(APs.items(), key=operator.itemgetter(1), reverse=True))

    #get overall AP percentage value
    AP = np.mean(list(APs.values()))

    if show_result:
        '''
         Draw MS COCO AP plot
        '''
        os.makedirs('result', exist_ok=True)
        window_title = "MSCOCO AP on different IOU"
        plot_title = "COCO AP = {0:.2f}%".format(AP)
        x_label = "Average Precision"
        output_path = os.path.join('result','COCO_AP.png')
        draw_plot_func(APs, len(APs), window_title, plot_title, x_label, output_path, to_show=False, plot_color='royalblue', true_p_bar='')

        print('\nMS COCO AP evaluation')
        for (iou_threshold, AP_value) in APs.items():
            print('IOU %.2f: AP %f' % (iou_threshold, AP_value))
        print('total AP: %f' % (AP))

    #return AP percentage value
    return AP, APs


def compute_AP_COCO_Scale(annotation_records, scale_gt_classes_records, pred_classes_records, class_names):
    '''
    Compute MSCOCO AP on different scale object: small, medium, large
    '''
    scale_APs = {}
    for scale_key in ['small','medium','large']:
        gt_classes_records = scale_gt_classes_records[scale_key]
        scale_AP, _ = compute_AP_COCO(annotation_records, gt_classes_records, pred_classes_records, class_names, show_result=False)
        scale_APs[scale_key] = round(scale_AP, 4)

    #get overall AP percentage value
    scale_mAP = np.mean(list(scale_APs.values()))

    '''
     Draw Scale AP plot
    '''
    os.makedirs('result', exist_ok=True)
    window_title = "MSCOCO AP on different scale"
    plot_title = "scale mAP = {0:.2f}%".format(scale_mAP)
    x_label = "Average Precision"
    output_path = os.path.join('result','COCO_scale_AP.png')
    draw_plot_func(scale_APs, len(scale_APs), window_title, plot_title, x_label, output_path, to_show=False, plot_color='royalblue', true_p_bar='')

    '''
     Draw Scale Object Sum plot
    '''
    for scale_key in ['small','medium','large']:
        gt_classes_records = scale_gt_classes_records[scale_key]
        gt_classes_sum = {}

        for _, class_name in enumerate(class_names):
            # summarize the gt object number for every class on different scale
            gt_classes_sum[class_name] = np.sum(len(gt_classes_records[class_name])) if class_name in gt_classes_records else 0

        total_sum = np.sum(list(gt_classes_sum.values()))

        window_title = "{} object number".format(scale_key)
        plot_title = "total {} object number = {}".format(scale_key, total_sum)
        x_label = "Object Number"
        output_path = os.path.join('result','{}_object_number.png'.format(scale_key))
        draw_plot_func(gt_classes_sum, len(gt_classes_sum), window_title, plot_title, x_label, output_path, to_show=False, plot_color='royalblue', true_p_bar='')

    print('\nMS COCO AP evaluation on different scale')
    for (scale, AP_value) in scale_APs.items():
        print('%s scale: AP %f' % (scale, AP_value))
    print('total AP: %f' % (scale_mAP))


def add_gt_record(gt_records, gt_record, class_name):
    # append or add ground truth class item
    if class_name in gt_records:
        gt_records[class_name].append(gt_record)
    else:
        gt_records[class_name] = list([gt_record])

    return gt_records


def get_scale_gt_dict(gt_classes_records, class_names):
    '''
    Get ground truth class dict on different object scales, according to MS COCO metrics definition:
        small objects: area < 32^2
        medium objects: 32^2 < area < 96^2
        large objects: area > 96^2
    input gt_classes_records would be like:
    gt_classes_records = {
        'car': [
                ['000001.jpg','100,120,200,235'],
                ['000002.jpg','85,63,156,128'],
                ...
               ],
        ...
    }
    return a record dict with following format, for AP/AR eval on different scale:
        scale_gt_classes_records = {
            'small': {
                'car': [
                        ['000001.jpg','100,120,200,235'],
                        ['000002.jpg','85,63,156,128'],
                        ...
                       ],
                ...
            },
            'medium': {
                'car': [
                        ['000003.jpg','100,120,200,235'],
                        ['000004.jpg','85,63,156,128'],
                        ...
                       ],
                ...
            },
            'large': {
                'car': [
                        ['000005.jpg','100,120,200,235'],
                        ['000006.jpg','85,63,156,128'],
                        ...
                       ],
                ...
            }
        }
    '''
    scale_gt_classes_records = {}
    small_gt_records = {}
    medium_gt_records = {}
    large_gt_records = {}

    for _, class_name in enumerate(class_names):
        gt_records = gt_classes_records[class_name]

        for (image_file, box) in gt_records:
            # get box area based on coordinate
            box_coord = [int(p) for p in box.split(',')]
            box_area = (box_coord[2] - box_coord[0]) * (box_coord[3] - box_coord[1])

            # add to corresponding gt records dict according to area size
            if box_area <= 32*32:
                small_gt_records = add_gt_record(small_gt_records, [image_file, box], class_name)
            elif box_area > 32*32 and box_area <= 96*96:
                medium_gt_records = add_gt_record(medium_gt_records, [image_file, box], class_name)
            elif box_area > 96*96:
                large_gt_records = add_gt_record(large_gt_records, [image_file, box], class_name)

    # form up scale_gt_classes_records
    scale_gt_classes_records['small'] = small_gt_records
    scale_gt_classes_records['medium'] = medium_gt_records
    scale_gt_classes_records['large'] = large_gt_records

    return scale_gt_classes_records


def get_filter_class_mAP(APs, class_filter, show_result=True):
    filtered_mAP = 0.0
    filtered_APs = OrderedDict()

    for (class_name, AP) in APs.items():
        if class_name in class_filter:
            filtered_APs[class_name] = AP

    filtered_mAP = np.mean(list(filtered_APs.values()))*100

    if show_result:
        print('\nfiltered classes AP')
        for (class_name, AP) in filtered_APs.items():
            print('%s: AP %.4f' % (class_name, AP))
        print('mAP:', filtered_mAP, '\n')
    return filtered_mAP


def eval_AP(model, model_format, annotation_lines, anchors, class_names, model_image_size, eval_type, iou_threshold, conf_threshold, elim_grid_sense, v5_decode, save_result, class_filter=None):
    '''
    Compute AP for detection model on annotation dataset
    '''
    annotation_records, gt_classes_records = annotation_parse(annotation_lines, class_names)
    pred_classes_records = get_prediction_class_records(model, model_format, annotation_records, anchors, class_names, model_image_size, conf_threshold, elim_grid_sense, v5_decode, save_result)
    AP = 0.0

    if eval_type == 'VOC':
        AP, APs = compute_mAP_PascalVOC(annotation_records, gt_classes_records, pred_classes_records, class_names, iou_threshold)

        if class_filter is not None:
            get_filter_class_mAP(APs, class_filter)

    elif eval_type == 'COCO':
        AP, _ = compute_AP_COCO(annotation_records, gt_classes_records, pred_classes_records, class_names, class_filter)
        # get AP for different scale: small, medium, large
        scale_gt_classes_records = get_scale_gt_dict(gt_classes_records, class_names)
        compute_AP_COCO_Scale(annotation_records, scale_gt_classes_records, pred_classes_records, class_names)
    else:
        raise ValueError('Unsupported evaluation type')

    return AP


#load TF 1.x frozen pb graph
def load_graph(model_path):
    # We parse the graph_def file
    with tf.gfile.GFile(model_path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # We load the graph_def in the default graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="graph",
            op_dict=None,
            producer_op_list=None
        )
    return graph


def load_eval_model(model_path):
    # support of tflite model
    if model_path.endswith('.tflite'):
        from tensorflow.lite.python import interpreter as interpreter_wrapper
        model = interpreter_wrapper.Interpreter(model_path=model_path)
        model.allocate_tensors()
        model_format = 'TFLITE'

    # support of MNN model
    elif model_path.endswith('.mnn'):
        model = MNN.Interpreter(model_path)
        model_format = 'MNN'

    # support of TF 1.x frozen pb model
    elif model_path.endswith('.pb'):
        model = load_graph(model_path)
        model_format = 'PB'

    # support of ONNX model
    elif model_path.endswith('.onnx'):
        model = onnxruntime.InferenceSession(model_path)
        model_format = 'ONNX'

    # normal keras h5 model
    elif model_path.endswith('.h5'):
        custom_object_dict = get_custom_objects()

        model = load_model(model_path, compile=False, custom_objects=custom_object_dict)
        model_format = 'H5'
        K.set_learning_phase(0)
    else:
        raise ValueError('invalid model file')

    return model, model_format


def main():
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description='evaluate YOLO model (h5/pb/onnx/tflite/mnn) with test dataset')
    '''
    Command line options
    '''
    parser.add_argument(
        '--model_path', type=str, required=True,
        help='path to model file')

    parser.add_argument(
        '--anchors_path', type=str, required=True,
        help='path to anchor definitions')

    parser.add_argument(
        '--classes_path', type=str, required=False,
        help='path to class definitions, default=%(default)s', default=os.path.join('configs' , 'voc_classes.txt'))

    parser.add_argument(
        '--classes_filter_path', type=str, required=False,
        help='path to class filter definitions, default=%(default)s', default=None)

    parser.add_argument(
        '--annotation_file', type=str, required=True,
        help='test annotation txt file')

    parser.add_argument(
        '--eval_type', type=str, required=False, choices=['VOC', 'COCO'],
        help='evaluation type (VOC/COCO), default=%(default)s', default='VOC')

    parser.add_argument(
        '--iou_threshold', type=float,
        help='IOU threshold for PascalVOC mAP, default=%(default)s', default=0.5)

    parser.add_argument(
        '--conf_threshold', type=float,
        help='confidence threshold for filtering box in postprocess, default=%(default)s', default=0.001)

    parser.add_argument(
        '--model_image_size', type=str,
        help='model image input size as <height>x<width>, default=%(default)s', default='416x416')

    parser.add_argument(
        '--elim_grid_sense', default=False, action="store_true",
        help = "Eliminate grid sensitivity")

    parser.add_argument(
        '--v5_decode', default=False, action="store_true",
        help = "Use YOLOv5 prediction decode")

    parser.add_argument(
        '--save_result', default=False, action="store_true",
        help='Save the detection result image in result/detection dir'
    )

    args = parser.parse_args()

    # param parse
    anchors = get_anchors(args.anchors_path)
    class_names = get_classes(args.classes_path)
    height, width = args.model_image_size.split('x')
    model_image_size = (int(height), int(width))
    assert (model_image_size[0]%32 == 0 and model_image_size[1]%32 == 0), 'model_image_size should be multiples of 32'

    # class filter parse
    if args.classes_filter_path is not None:
        class_filter = get_classes(args.classes_filter_path)
    else:
        class_filter = None

    annotation_lines = get_dataset(args.annotation_file, shuffle=False)
    model, model_format = load_eval_model(args.model_path)

    start = time.time()
    eval_AP(model, model_format, annotation_lines, anchors, class_names, model_image_size, args.eval_type, args.iou_threshold, args.conf_threshold, args.elim_grid_sense, args.v5_decode, args.save_result, class_filter=class_filter)
    end = time.time()
    print("Evaluation time cost: {:.6f}s".format(end - start))


if __name__ == '__main__':
    main()
