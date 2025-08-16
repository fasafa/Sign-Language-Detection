# Sign-Language-Detection

# **Install YOLO11 via Ultralytics**


```python
!pip install ultralytics supervision roboflow

from IPython import display
display.clear_output()

!pip install ultralytics --quiet

```


```python
import ultralytics
ultralytics.checks()
```

    Ultralytics 8.3.179 🚀 Python-3.11.13 torch-2.6.0+cu124 CUDA:0 (Tesla T4, 15095MiB)
    Setup complete ✅ (2 CPUs, 12.7 GB RAM, 43.0/112.6 GB disk)


# **Load the Dataset**


```python
!unzip /content/ASL.v1i.yolov11.zip -d /content/dataset
```

        

## **Training the YOLO model**


```python
!yolo task=detect mode=train model=yolo11n.pt data=/content/dataset/data.yaml epochs=500 imgsz=640 plots=True

```

    
Downloading https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt to 'yolo11n.pt':   0% 0.00/5.35M [00:00<?, ?B/s]
Downloading https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt to 'yolo11n.pt': 100% 5.35M/5.35M [00:00<00:00, 130MB/s]
    Ultralytics 8.3.179 🚀 Python-3.11.13 torch-2.6.0+cu124 CUDA:0 (Tesla T4, 15095MiB)
    [34m[1mengine/trainer: [0magnostic_nms=False, amp=True, augment=False, auto_augment=randaugment, batch=16, bgr=0.0, box=7.5, cache=False, cfg=None, classes=None, close_mosaic=10, cls=0.5, conf=None, copy_paste=0.0, copy_paste_mode=flip, cos_lr=False, cutmix=0.0, data=/content/dataset/data.yaml, degrees=0.0, deterministic=True, device=None, dfl=1.5, dnn=False, dropout=0.0, dynamic=False, embed=None, epochs=500, erasing=0.4, exist_ok=False, fliplr=0.5, flipud=0.0, format=torchscript, fraction=1.0, freeze=None, half=False, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, imgsz=640, int8=False, iou=0.7, keras=False, kobj=1.0, line_width=None, lr0=0.01, lrf=0.01, mask_ratio=4, max_det=300, mixup=0.0, mode=train, model=yolo11n.pt, momentum=0.937, mosaic=1.0, multi_scale=False, name=train, nbs=64, nms=False, opset=None, optimize=False, optimizer=auto, overlap_mask=True, patience=100, perspective=0.0, plots=True, pose=12.0, pretrained=True, profile=False, project=None, rect=False, resume=False, retina_masks=False, save=True, save_conf=False, save_crop=False, save_dir=runs/detect/train, save_frames=False, save_json=False, save_period=-1, save_txt=False, scale=0.5, seed=0, shear=0.0, show=False, show_boxes=True, show_conf=True, show_labels=True, simplify=True, single_cls=False, source=None, split=val, stream_buffer=False, task=detect, time=None, tracker=botsort.yaml, translate=0.1, val=True, verbose=True, vid_stride=1, visualize=False, warmup_bias_lr=0.1, warmup_epochs=3.0, warmup_momentum=0.8, weight_decay=0.0005, workers=8, workspace=None
    Downloading https://ultralytics.com/assets/Arial.ttf to '/root/.config/Ultralytics/Arial.ttf': 100% 755k/755k [00:00<00:00, 57.0MB/s]
    Overriding model.yaml nc=80 with nc=26
    
                       from  n    params  module                                       arguments                     
      0                  -1  1       464  ultralytics.nn.modules.conv.Conv             [3, 16, 3, 2]                 
      1                  -1  1      4672  ultralytics.nn.modules.conv.Conv             [16, 32, 3, 2]                
      2                  -1  1      6640  ultralytics.nn.modules.block.C3k2            [32, 64, 1, False, 0.25]      
      3                  -1  1     36992  ultralytics.nn.modules.conv.Conv             [64, 64, 3, 2]                
      4                  -1  1     26080  ultralytics.nn.modules.block.C3k2            [64, 128, 1, False, 0.25]     
      5                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]              
      6                  -1  1     87040  ultralytics.nn.modules.block.C3k2            [128, 128, 1, True]           
      7                  -1  1    295424  ultralytics.nn.modules.conv.Conv             [128, 256, 3, 2]              
      8                  -1  1    346112  ultralytics.nn.modules.block.C3k2            [256, 256, 1, True]           
      9                  -1  1    164608  ultralytics.nn.modules.block.SPPF            [256, 256, 5]                 
     10                  -1  1    249728  ultralytics.nn.modules.block.C2PSA           [256, 256, 1]                 
     11                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
     12             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
     13                  -1  1    111296  ultralytics.nn.modules.block.C3k2            [384, 128, 1, False]          
     14                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
     15             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
     16                  -1  1     32096  ultralytics.nn.modules.block.C3k2            [256, 64, 1, False]           
     17                  -1  1     36992  ultralytics.nn.modules.conv.Conv             [64, 64, 3, 2]                
     18            [-1, 13]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
     19                  -1  1     86720  ultralytics.nn.modules.block.C3k2            [192, 128, 1, False]          
     20                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]              
     21            [-1, 10]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
     22                  -1  1    378880  ultralytics.nn.modules.block.C3k2            [384, 256, 1, True]           
     23        [16, 19, 22]  1    435742  ultralytics.nn.modules.head.Detect           [26, [64, 128, 256]]          
    YOLO11n summary: 181 layers, 2,594,910 parameters, 2,594,894 gradients, 6.5 GFLOPs
    
    Transferred 448/499 items from pretrained weights
    Freezing layer 'model.23.dfl.conv.weight'
    [34m[1mAMP: [0mrunning Automatic Mixed Precision (AMP) checks...
    [34m[1mAMP: [0mchecks passed ✅
    [34m[1mtrain: [0mFast image access ✅ (ping: 0.0±0.0 ms, read: 584.6±187.9 MB/s, size: 26.5 KB)
    [34m[1mtrain: [0mScanning /content/dataset/train/labels... 996 images, 0 backgrounds, 0 corrupt: 100% 996/996 [00:00<00:00, 2222.81it/s]
    [34m[1mtrain: [0mNew cache created: /content/dataset/train/labels.cache
    [34m[1malbumentations: [0mBlur(p=0.01, blur_limit=(3, 7)), MedianBlur(p=0.01, blur_limit=(3, 7)), ToGray(p=0.01, method='weighted_average', num_output_channels=3), CLAHE(p=0.01, clip_limit=(1.0, 4.0), tile_grid_size=(8, 8))
    [34m[1mval: [0mFast image access ✅ (ping: 0.0±0.0 ms, read: 413.7±184.0 MB/s, size: 26.9 KB)
    [34m[1mval: [0mScanning /content/dataset/valid/labels... 141 images, 0 backgrounds, 0 corrupt: 100% 141/141 [00:00<00:00, 1408.63it/s]
    [34m[1mval: [0mNew cache created: /content/dataset/valid/labels.cache
    Plotting labels to runs/detect/train/labels.jpg... 
    [34m[1moptimizer:[0m 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... 
    [34m[1moptimizer:[0m AdamW(lr=0.000333, momentum=0.9) with parameter groups 81 weight(decay=0.0), 88 weight(decay=0.0005), 87 bias(decay=0.0)
    Image sizes 640 train, 640 val
    Using 2 dataloader workers
    Logging results to [1mruns/detect/train[0m
    Starting training for 500 epochs...
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          1/500      2.44G      1.093      4.329      1.455          9        640: 100% 63/63 [00:21<00:00,  2.98it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  2.51it/s]
                       all        141        141     0.0326       0.38     0.0811     0.0704
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          2/500      2.59G     0.9392      3.889      1.318         12        640: 100% 63/63 [00:17<00:00,  3.67it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.60it/s]
                       all        141        141      0.493       0.23       0.22      0.175
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          3/500      2.61G       1.01      3.489      1.357          8        640: 100% 63/63 [00:18<00:00,  3.36it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.21it/s]
                       all        141        141      0.447      0.346       0.39      0.307
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          4/500      2.62G      1.042      3.109      1.383          8        640: 100% 63/63 [00:17<00:00,  3.70it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.29it/s]
                       all        141        141      0.331      0.641      0.541      0.433
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          5/500      2.64G     0.9595      2.756      1.311         10        640: 100% 63/63 [00:18<00:00,  3.47it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.76it/s]
                       all        141        141      0.564      0.585       0.67       0.58
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          6/500      2.64G     0.9112       2.53      1.265          5        640: 100% 63/63 [00:16<00:00,  3.74it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.45it/s]
                       all        141        141      0.549      0.727      0.712      0.619
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          7/500      2.67G     0.8683      2.302      1.235         12        640: 100% 63/63 [00:17<00:00,  3.66it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  2.90it/s]
                       all        141        141      0.503      0.701       0.71      0.646
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          8/500      2.68G     0.8558      2.157      1.234          9        640: 100% 63/63 [00:16<00:00,  3.81it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.78it/s]
                       all        141        141      0.651       0.71      0.776        0.7
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          9/500       2.7G     0.8025          2      1.173         16        640: 100% 63/63 [00:16<00:00,  3.86it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.36it/s]
                       all        141        141      0.723      0.785      0.817      0.734
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
         10/500       2.7G     0.8118      1.913      1.192          7        640: 100% 63/63 [00:17<00:00,  3.63it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.16it/s]
                       all        141        141      0.745      0.753      0.841      0.745
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
         11/500      2.73G     0.7782      1.778      1.162         12        640: 100% 63/63 [00:16<00:00,  3.78it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.29it/s]
                       all        141        141      0.765      0.742      0.842      0.765
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
         12/500      2.74G     0.7687      1.654      1.135         13        640: 100% 63/63 [00:17<00:00,  3.58it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.01it/s]
                       all        141        141      0.757       0.84      0.855       0.78
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
         13/500      2.76G     0.7481       1.63      1.144         11        640: 100% 63/63 [00:16<00:00,  3.83it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.84it/s]
                       all        141        141      0.857      0.803      0.892      0.807
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
         14/500      2.76G     0.7664      1.544      1.144         15        640: 100% 63/63 [00:17<00:00,  3.62it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  3.30it/s]
                       all        141        141      0.753      0.767       0.87      0.802
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
         15/500      2.79G     0.7475      1.479      1.141          9        640: 100% 63/63 [00:16<00:00,  3.78it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.64it/s]
                       all        141        141      0.869       0.85      0.916      0.846
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
         16/500       2.8G     0.7124       1.39      1.115          8        640: 100% 63/63 [00:16<00:00,  3.72it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.19it/s]
                       all        141        141      0.878      0.836      0.908      0.846
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
         17/500      2.81G     0.7189      1.412      1.121          8        640: 100% 63/63 [00:17<00:00,  3.62it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.38it/s]
                       all        141        141      0.822      0.839      0.908      0.837
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
         18/500      2.82G     0.7062       1.32      1.106          9        640: 100% 63/63 [00:16<00:00,  3.80it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.63it/s]
                       all        141        141       0.86      0.852      0.939      0.868
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
         19/500      2.84G     0.6907      1.268      1.099          9        640: 100% 63/63 [00:17<00:00,  3.51it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.07it/s]
                       all        141        141      0.822      0.859      0.917      0.856
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
         20/500      2.86G     0.6858      1.251      1.098          8        640: 100% 63/63 [00:16<00:00,  3.86it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.44it/s]
                       all        141        141      0.858      0.868      0.937      0.872
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
         21/500      2.87G     0.6987      1.211      1.095          6        640: 100% 63/63 [00:17<00:00,  3.62it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  3.32it/s]
                       all        141        141      0.867       0.84      0.946      0.884
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
         22/500      2.88G     0.6826      1.157      1.086          6        640: 100% 63/63 [00:16<00:00,  3.85it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.02it/s]
                       all        141        141      0.895      0.805      0.924      0.854
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
         23/500       2.9G     0.6728      1.145      1.083         11        640: 100% 63/63 [00:16<00:00,  3.73it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.67it/s]
                       all        141        141      0.871       0.85      0.924      0.848
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
         24/500      2.92G     0.6503      1.117      1.073          9        640: 100% 63/63 [00:17<00:00,  3.61it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.90it/s]
                       all        141        141      0.905      0.878      0.927      0.858
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
         25/500      2.93G     0.6724      1.096      1.086         11        640: 100% 63/63 [00:16<00:00,  3.79it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.65it/s]
                       all        141        141      0.799       0.84      0.882      0.817
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
         26/500      2.94G     0.6612      1.061      1.074         10        640: 100% 63/63 [00:17<00:00,  3.61it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.40it/s]
                       all        141        141      0.883      0.863      0.947      0.881
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
         27/500      2.96G     0.6384      1.026      1.067          9        640: 100% 63/63 [00:16<00:00,  3.77it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.23it/s]
                       all        141        141      0.914      0.889      0.952      0.887
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
         28/500      2.98G     0.6324       1.04       1.08          4        640: 100% 63/63 [00:16<00:00,  3.71it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  3.07it/s]
                       all        141        141      0.832       0.86      0.942      0.877
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
         29/500      2.99G     0.6286      1.007      1.061          8        640: 100% 63/63 [00:16<00:00,  3.75it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.46it/s]
                       all        141        141      0.921       0.87       0.96      0.903
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
         30/500         3G     0.6143     0.9852      1.047         10        640: 100% 63/63 [00:16<00:00,  3.79it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.09it/s]
                       all        141        141      0.912      0.861      0.964      0.896
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
         31/500      3.02G     0.6488     0.9975      1.085          9        640: 100% 63/63 [00:17<00:00,  3.61it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.65it/s]
                       all        141        141      0.865      0.852      0.928      0.858
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
         32/500      3.04G     0.6218     0.9128      1.052         10        640: 100% 63/63 [00:16<00:00,  3.80it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.01it/s]
                       all        141        141       0.85      0.813      0.941      0.882
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
         33/500      3.05G     0.6207     0.9296      1.056          8        640: 100% 63/63 [00:17<00:00,  3.55it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.75it/s]
                       all        141        141      0.906      0.867      0.957      0.892
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
         34/500      3.06G     0.6262      0.905      1.061         10        640: 100% 63/63 [00:16<00:00,  3.82it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.51it/s]
                       all        141        141      0.855      0.898      0.953      0.879
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
         35/500      3.08G      0.602     0.8845      1.052         10        640: 100% 63/63 [00:17<00:00,  3.68it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  3.27it/s]
                       all        141        141      0.879      0.886      0.941      0.881
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
         36/500       3.1G      0.609     0.8443       1.05         10        640: 100% 63/63 [00:16<00:00,  3.72it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.56it/s]
                       all        141        141        0.9      0.873       0.92      0.866
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
         37/500      3.11G     0.6121     0.8703      1.059          8        640: 100% 63/63 [00:16<00:00,  3.75it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.79it/s]
                       all        141        141      0.846      0.866      0.936      0.879
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
         38/500      3.12G     0.6073      0.836       1.04          8        640: 100% 63/63 [00:17<00:00,  3.65it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.30it/s]
                       all        141        141      0.856      0.874      0.947      0.875
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
         39/500      3.14G     0.6226     0.8606      1.052          9        640: 100% 63/63 [00:16<00:00,  3.74it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.54it/s]
                       all        141        141      0.876      0.865      0.947      0.877
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
         40/500      3.16G     0.6086     0.8222      1.054         14        640: 100% 63/63 [00:18<00:00,  3.47it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.13it/s]
                       all        141        141      0.911      0.874      0.956      0.897
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
         41/500      3.17G     0.5782     0.8162      1.029         12        640: 100% 63/63 [00:17<00:00,  3.63it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.62it/s]
                       all        141        141      0.899      0.914      0.959      0.901
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
         42/500      3.18G     0.6165     0.8231      1.065         10        640: 100% 63/63 [00:18<00:00,  3.45it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.07it/s]
                       all        141        141      0.908      0.844      0.942      0.879
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
         43/500       3.2G     0.5756     0.7921      1.031          7        640: 100% 63/63 [00:16<00:00,  3.76it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.15it/s]
                       all        141        141      0.907      0.865      0.955      0.895
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
         44/500      3.21G     0.5888     0.7737      1.035         12        640: 100% 63/63 [00:16<00:00,  3.76it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  2.94it/s]
                       all        141        141      0.865      0.851      0.931      0.867
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
         45/500      3.23G     0.5863      0.759      1.047          8        640: 100% 63/63 [00:17<00:00,  3.63it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.10it/s]
                       all        141        141      0.852      0.892      0.951      0.884
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
         46/500      3.24G     0.5735     0.7516      1.025          9        640: 100% 63/63 [00:17<00:00,  3.70it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  3.92it/s]
                       all        141        141      0.879      0.919      0.958        0.9
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
         47/500      3.26G     0.5756     0.7641      1.026         12        640: 100% 63/63 [00:17<00:00,  3.58it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.10it/s]
                       all        141        141      0.896      0.842      0.939      0.868
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
         48/500      3.28G      0.569     0.7419      1.036         14        640: 100% 63/63 [00:17<00:00,  3.64it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.13it/s]
                       all        141        141      0.898      0.864      0.937      0.869
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
         49/500      3.29G     0.5671     0.7572      1.025         12        640: 100% 63/63 [00:18<00:00,  3.45it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.32it/s]
                       all        141        141      0.891       0.87      0.954      0.886
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
         50/500       3.3G     0.5691     0.7125      1.024          9        640: 100% 63/63 [00:17<00:00,  3.63it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.87it/s]
                       all        141        141      0.934      0.927      0.972      0.912
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
         51/500      3.32G     0.5595     0.7001      1.022         11        640: 100% 63/63 [00:18<00:00,  3.41it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.52it/s]
                       all        141        141      0.878      0.909      0.965        0.9
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
         52/500      3.33G     0.5748     0.7167       1.03          8        640: 100% 63/63 [00:16<00:00,  3.73it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.45it/s]
                       all        141        141      0.923      0.883      0.947      0.885
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
         53/500      3.35G     0.5487      0.675       1.01         10        640: 100% 63/63 [00:17<00:00,  3.53it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.14it/s]
                       all        141        141      0.905      0.885      0.957      0.891
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
         54/500      3.36G     0.5625     0.6799      1.022          8        640: 100% 63/63 [00:16<00:00,  3.72it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.95it/s]
                       all        141        141      0.906       0.87      0.943      0.891
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
         55/500      3.38G     0.5401      0.658      1.009          8        640: 100% 63/63 [00:17<00:00,  3.59it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  3.45it/s]
                       all        141        141      0.907      0.865      0.947      0.886
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
         56/500      3.39G     0.5503     0.6782      1.011         10        640: 100% 63/63 [00:17<00:00,  3.70it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.61it/s]
                       all        141        141      0.898       0.86      0.948      0.882
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
         57/500      3.41G     0.5608     0.6622      1.022         10        640: 100% 63/63 [00:17<00:00,  3.68it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  3.55it/s]
                       all        141        141      0.898      0.875      0.938      0.869
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
         58/500      3.42G     0.5698     0.6624      1.027          7        640: 100% 63/63 [00:17<00:00,  3.57it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.49it/s]
                       all        141        141      0.888      0.873       0.94       0.87
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
         59/500      3.44G     0.5636     0.6645      1.033         10        640: 100% 63/63 [00:17<00:00,  3.68it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.97it/s]
                       all        141        141      0.888       0.86      0.948      0.882
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
         60/500      3.45G      0.565     0.6442      1.024         11        640: 100% 63/63 [00:18<00:00,  3.46it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.71it/s]
                       all        141        141      0.927      0.867      0.947      0.881
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
         61/500      3.47G     0.5504     0.6512      1.018          8        640: 100% 63/63 [00:17<00:00,  3.61it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  3.99it/s]
                       all        141        141      0.914       0.84      0.958      0.893
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
         62/500      3.47G     0.5508     0.6552       1.02          7        640: 100% 63/63 [00:17<00:00,  3.50it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.28it/s]
                       all        141        141      0.915      0.851      0.924      0.862
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
         63/500       3.5G     0.5503     0.6412       1.02          8        640: 100% 63/63 [00:17<00:00,  3.54it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.67it/s]
                       all        141        141      0.911       0.89      0.955      0.884
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
         64/500      3.51G       0.54     0.6581      1.019          8        640: 100% 63/63 [00:18<00:00,  3.48it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.69it/s]
                       all        141        141      0.904      0.893      0.941      0.887
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
         65/500      3.53G     0.5316     0.6185      1.012         11        640: 100% 63/63 [00:17<00:00,  3.63it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.24it/s]
                       all        141        141      0.943      0.903      0.968      0.909
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
         66/500      3.53G     0.5314     0.6203      1.011         11        640: 100% 63/63 [00:18<00:00,  3.44it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.97it/s]
                       all        141        141      0.923      0.837      0.931      0.868
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
         67/500      3.56G     0.5292     0.6234      1.003         10        640: 100% 63/63 [00:17<00:00,  3.64it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.03it/s]
                       all        141        141      0.941      0.851      0.944      0.894
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
         68/500      3.57G     0.5412     0.6276      1.014          6        640: 100% 63/63 [00:18<00:00,  3.47it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.24it/s]
                       all        141        141      0.926      0.847      0.949      0.888
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
         69/500      3.59G      0.536     0.6171      1.007         11        640: 100% 63/63 [00:17<00:00,  3.68it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.12it/s]
                       all        141        141      0.906      0.876      0.951      0.893
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
         70/500      3.59G     0.5309     0.5821      1.015         13        640: 100% 63/63 [00:18<00:00,  3.50it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.42it/s]
                       all        141        141      0.872      0.879      0.944      0.887
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
         71/500      3.62G     0.5236     0.5765      1.005          8        640: 100% 63/63 [00:17<00:00,  3.70it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.83it/s]
                       all        141        141       0.91       0.84      0.939      0.881
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
         72/500      3.63G     0.5329      0.588     0.9963          9        640: 100% 63/63 [00:17<00:00,  3.56it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  3.46it/s]
                       all        141        141      0.909      0.831      0.946      0.885
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
         73/500      3.64G      0.511     0.5509     0.9968         10        640: 100% 63/63 [00:16<00:00,  3.73it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.88it/s]
                       all        141        141      0.922      0.868      0.944      0.888
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
         74/500      3.65G     0.5089     0.5705     0.9943         10        640: 100% 63/63 [00:17<00:00,  3.67it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  3.63it/s]
                       all        141        141      0.902      0.852      0.931      0.885
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
         75/500      3.68G     0.5278     0.5889      1.011         11        640: 100% 63/63 [00:17<00:00,  3.70it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.53it/s]
                       all        141        141      0.902      0.872      0.932      0.874
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
         76/500      3.69G     0.5279      0.581      1.005          9        640: 100% 63/63 [00:16<00:00,  3.73it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.77it/s]
                       all        141        141      0.902      0.894      0.964      0.901
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
         77/500      3.71G     0.5213     0.5675      1.004         10        640: 100% 63/63 [00:18<00:00,  3.43it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.88it/s]
                       all        141        141      0.926      0.891      0.956      0.896
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
         78/500      3.71G      0.508      0.548     0.9924         14        640: 100% 63/63 [00:16<00:00,  3.72it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.55it/s]
                       all        141        141      0.898      0.875      0.937      0.883
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
         79/500      3.73G     0.5172     0.5509      1.001          7        640: 100% 63/63 [00:18<00:00,  3.45it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.13it/s]
                       all        141        141      0.949      0.884      0.959      0.907
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
         80/500      3.75G     0.5043     0.5277     0.9968          8        640: 100% 63/63 [00:16<00:00,  3.71it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.01it/s]
                       all        141        141      0.866      0.907      0.948      0.887
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
         81/500      3.76G     0.5157     0.5697      1.002         16        640: 100% 63/63 [00:18<00:00,  3.49it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.30it/s]
                       all        141        141      0.918      0.895      0.952      0.893
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
         82/500      3.77G     0.5084     0.5448     0.9952         12        640: 100% 63/63 [00:17<00:00,  3.59it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.47it/s]
                       all        141        141      0.879      0.864       0.95      0.891
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
         83/500      3.79G     0.4972     0.5385      0.991         11        640: 100% 63/63 [00:17<00:00,  3.57it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  3.85it/s]
                       all        141        141      0.925      0.866      0.952      0.898
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
         84/500      3.81G     0.4954     0.5423     0.9896         11        640: 100% 63/63 [00:17<00:00,  3.65it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.08it/s]
                       all        141        141      0.926      0.865      0.952      0.896
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
         85/500      3.82G     0.4863     0.5258     0.9812         12        640: 100% 63/63 [00:17<00:00,  3.66it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  3.18it/s]
                       all        141        141      0.933      0.883      0.951      0.898
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
         86/500      3.83G     0.4994     0.5387     0.9994          9        640: 100% 63/63 [00:16<00:00,  3.75it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.05it/s]
                       all        141        141      0.871      0.904      0.949      0.893
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
         87/500      3.85G     0.4925      0.531     0.9796         12        640: 100% 63/63 [00:17<00:00,  3.68it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.04it/s]
                       all        141        141      0.907       0.83      0.934      0.886
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
         88/500      3.87G     0.4818      0.528     0.9842          8        640: 100% 63/63 [00:17<00:00,  3.69it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.31it/s]
                       all        141        141      0.943      0.888      0.957      0.905
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
         89/500      3.88G     0.4867     0.5278     0.9857         10        640: 100% 63/63 [00:16<00:00,  3.73it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.91it/s]
                       all        141        141      0.883      0.844      0.931      0.877
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
         90/500      3.89G     0.5111      0.546     0.9947         11        640: 100% 63/63 [00:17<00:00,  3.53it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.64it/s]
                       all        141        141      0.894      0.855      0.933      0.879
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
         91/500      3.91G      0.489     0.5263      0.984         11        640: 100% 63/63 [00:16<00:00,  3.74it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.32it/s]
                       all        141        141      0.963      0.892      0.971      0.917
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
         92/500      3.93G     0.4959     0.5119     0.9958          9        640: 100% 63/63 [00:17<00:00,  3.64it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  3.51it/s]
                       all        141        141      0.952       0.88      0.965      0.914
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
         93/500      3.94G     0.4777     0.4978     0.9804         11        640: 100% 63/63 [00:16<00:00,  3.72it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.26it/s]
                       all        141        141      0.912      0.851      0.951      0.895
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
         94/500      3.95G     0.4826     0.5096     0.9873          6        640: 100% 63/63 [00:17<00:00,  3.55it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  3.23it/s]
                       all        141        141      0.898      0.881      0.943      0.891
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
         95/500      3.97G     0.4615     0.4922     0.9817          9        640: 100% 63/63 [00:17<00:00,  3.64it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.20it/s]
                       all        141        141       0.91      0.865      0.948      0.888
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
         96/500      3.99G     0.4835     0.5034     0.9855         10        640: 100% 63/63 [00:16<00:00,  3.73it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  2.97it/s]
                       all        141        141      0.927      0.874      0.942      0.889
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
         97/500         4G     0.4831     0.5058     0.9931          8        640: 100% 63/63 [00:17<00:00,  3.70it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.08it/s]
                       all        141        141      0.939      0.884      0.947      0.893
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
         98/500      4.01G     0.4897     0.5164     0.9937         11        640: 100% 63/63 [00:16<00:00,  3.72it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.39it/s]
                       all        141        141      0.898      0.852       0.94      0.891
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
         99/500      4.03G     0.4812     0.4838     0.9865          9        640: 100% 63/63 [00:18<00:00,  3.47it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.60it/s]
                       all        141        141        0.9      0.913      0.956      0.899
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        100/500      4.04G     0.4682     0.4812     0.9732         11        640: 100% 63/63 [00:17<00:00,  3.66it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.11it/s]
                       all        141        141      0.923      0.869      0.946      0.901
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        101/500      4.06G     0.4883     0.4776     0.9897          8        640: 100% 63/63 [00:18<00:00,  3.40it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.86it/s]
                       all        141        141      0.872      0.897      0.938      0.881
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        102/500      4.07G     0.4841     0.4972      0.979          8        640: 100% 63/63 [00:17<00:00,  3.55it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.20it/s]
                       all        141        141      0.927      0.893      0.957      0.897
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        103/500      4.09G     0.4765     0.4743     0.9762          7        640: 100% 63/63 [00:18<00:00,  3.34it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.68it/s]
                       all        141        141      0.938       0.88      0.951      0.894
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        104/500       4.1G     0.4784     0.4863     0.9765         10        640: 100% 63/63 [00:17<00:00,  3.62it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.32it/s]
                       all        141        141       0.92      0.875      0.943       0.89
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        105/500      4.12G     0.4592     0.4638     0.9736          4        640: 100% 63/63 [00:18<00:00,  3.36it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.06it/s]
                       all        141        141      0.947      0.881      0.964      0.909
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        106/500      4.13G     0.4835     0.4878     0.9792         13        640: 100% 63/63 [00:17<00:00,  3.63it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.15it/s]
                       all        141        141      0.874       0.91      0.967      0.915
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        107/500      4.15G     0.4804     0.4634     0.9736         11        640: 100% 63/63 [00:18<00:00,  3.40it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.65it/s]
                       all        141        141      0.939      0.873      0.952      0.892
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        108/500      4.16G     0.4566     0.4661     0.9652          9        640: 100% 63/63 [00:17<00:00,  3.69it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.52it/s]
                       all        141        141      0.907      0.907      0.961      0.913
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        109/500      4.18G     0.4752     0.4652      0.989         10        640: 100% 63/63 [00:18<00:00,  3.49it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  3.57it/s]
                       all        141        141      0.914      0.863      0.964      0.912
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        110/500      4.19G     0.4633     0.4579     0.9776          9        640: 100% 63/63 [00:17<00:00,  3.65it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.68it/s]
                       all        141        141      0.895      0.875      0.946      0.886
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        111/500      4.21G     0.4814     0.4711      0.986          8        640: 100% 63/63 [00:17<00:00,  3.67it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  2.61it/s]
                       all        141        141      0.908      0.878      0.945      0.887
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        112/500      4.22G     0.4697     0.4714     0.9667         10        640: 100% 63/63 [00:17<00:00,  3.70it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.11it/s]
                       all        141        141       0.87      0.917      0.945      0.894
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        113/500      4.24G     0.4589     0.4551     0.9824          6        640: 100% 63/63 [00:17<00:00,  3.64it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.71it/s]
                       all        141        141      0.895      0.884      0.943      0.898
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        114/500      4.24G      0.459     0.4391     0.9801         10        640: 100% 63/63 [00:18<00:00,  3.47it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.05it/s]
                       all        141        141        0.9      0.897      0.948      0.893
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        115/500      4.27G     0.4517     0.4455     0.9688          9        640: 100% 63/63 [00:17<00:00,  3.60it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.33it/s]
                       all        141        141       0.89      0.875      0.935      0.877
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        116/500      4.28G     0.4644     0.4725      0.974          8        640: 100% 63/63 [00:18<00:00,  3.43it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.10it/s]
                       all        141        141      0.936      0.867      0.956      0.898
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        117/500       4.3G     0.4706     0.4678      0.982         12        640: 100% 63/63 [00:17<00:00,  3.62it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.65it/s]
                       all        141        141      0.942       0.87      0.941      0.884
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        118/500       4.3G     0.4529     0.4501      0.971          9        640: 100% 63/63 [00:18<00:00,  3.49it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.32it/s]
                       all        141        141      0.948       0.86      0.967      0.915
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        119/500      4.33G     0.4619     0.4336     0.9689          7        640: 100% 63/63 [00:17<00:00,  3.68it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.78it/s]
                       all        141        141      0.937      0.893      0.955      0.902
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        120/500      4.34G     0.4729     0.4449     0.9768         11        640: 100% 63/63 [00:18<00:00,  3.46it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.24it/s]
                       all        141        141      0.922      0.886      0.958      0.904
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        121/500      4.36G     0.4579     0.4366     0.9759          7        640: 100% 63/63 [00:16<00:00,  3.73it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.98it/s]
                       all        141        141      0.942      0.919      0.967      0.914
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        122/500      4.36G     0.4485     0.4471     0.9607          9        640: 100% 63/63 [00:18<00:00,  3.49it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.56it/s]
                       all        141        141      0.901      0.894      0.953      0.898
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        123/500      4.39G     0.4453     0.4358     0.9669         10        640: 100% 63/63 [00:16<00:00,  3.72it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.52it/s]
                       all        141        141      0.903      0.885      0.956      0.904
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        124/500       4.4G     0.4608     0.4512     0.9828          8        640: 100% 63/63 [00:17<00:00,  3.58it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  3.40it/s]
                       all        141        141      0.918      0.913      0.961      0.904
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        125/500      4.41G      0.456     0.4613     0.9753          8        640: 100% 63/63 [00:17<00:00,  3.63it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.28it/s]
                       all        141        141      0.937      0.882      0.964       0.91
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        126/500      4.42G      0.448     0.4269     0.9677          9        640: 100% 63/63 [00:17<00:00,  3.62it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  3.35it/s]
                       all        141        141      0.921      0.912      0.963      0.908
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        127/500      4.45G     0.4608     0.4496     0.9692          9        640: 100% 63/63 [00:17<00:00,  3.61it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.46it/s]
                       all        141        141      0.942      0.893      0.967      0.912
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        128/500      4.46G     0.4462     0.4334     0.9696         11        640: 100% 63/63 [00:17<00:00,  3.68it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.85it/s]
                       all        141        141       0.87      0.902      0.964      0.911
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        129/500      4.47G     0.4562     0.4264     0.9699          8        640: 100% 63/63 [00:17<00:00,  3.59it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.77it/s]
                       all        141        141      0.878      0.912       0.96      0.906
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        130/500      4.48G     0.4371     0.4262     0.9655          9        640: 100% 63/63 [00:17<00:00,  3.70it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.44it/s]
                       all        141        141      0.934      0.903      0.971      0.917
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        131/500       4.5G     0.4369      0.414     0.9641          9        640: 100% 63/63 [00:18<00:00,  3.45it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.24it/s]
                       all        141        141      0.929      0.912      0.977      0.927
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        132/500      4.52G     0.4337     0.4152      0.963          7        640: 100% 63/63 [00:16<00:00,  3.72it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.05it/s]
                       all        141        141      0.929      0.886      0.963      0.914
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        133/500      4.53G     0.4417     0.4237      0.967          9        640: 100% 63/63 [00:17<00:00,  3.50it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.72it/s]
                       all        141        141      0.899      0.894      0.948      0.895
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        134/500      4.54G     0.4564     0.4309     0.9644          6        640: 100% 63/63 [00:16<00:00,  3.71it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.06it/s]
                       all        141        141      0.954       0.91      0.969      0.919
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        135/500      4.56G     0.4297     0.4046     0.9582          7        640: 100% 63/63 [00:17<00:00,  3.60it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  3.24it/s]
                       all        141        141       0.94      0.894      0.965       0.91
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        136/500      4.58G     0.4444     0.4184     0.9711          6        640: 100% 63/63 [00:16<00:00,  3.86it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  3.83it/s]
                       all        141        141       0.94        0.9      0.952      0.902
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        137/500      4.59G     0.4372     0.4115     0.9603          7        640: 100% 63/63 [00:16<00:00,  3.77it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.74it/s]
                       all        141        141      0.922      0.891      0.946      0.892
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        138/500       4.6G     0.4297     0.4052     0.9622          6        640: 100% 63/63 [00:17<00:00,  3.59it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.70it/s]
                       all        141        141       0.93      0.881      0.958      0.895
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        139/500      4.62G     0.4415     0.4243     0.9683          9        640: 100% 63/63 [00:16<00:00,  3.77it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.81it/s]
                       all        141        141       0.94      0.878      0.947      0.895
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        140/500      4.64G     0.4417     0.4031     0.9567          6        640: 100% 63/63 [00:18<00:00,  3.50it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.32it/s]
                       all        141        141      0.949      0.886      0.956      0.901
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        141/500      4.65G     0.4442      0.414      0.965         14        640: 100% 63/63 [00:17<00:00,  3.69it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.85it/s]
                       all        141        141      0.938        0.9      0.969      0.913
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        142/500      4.66G      0.421     0.3854     0.9439         12        640: 100% 63/63 [00:17<00:00,  3.60it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.01it/s]
                       all        141        141      0.909      0.914      0.958      0.901
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        143/500      4.68G     0.4371     0.4161     0.9674          8        640: 100% 63/63 [00:17<00:00,  3.67it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.39it/s]
                       all        141        141      0.908      0.889       0.95      0.889
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        144/500       4.7G     0.4292     0.4044     0.9546         13        640: 100% 63/63 [00:16<00:00,  3.72it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  2.94it/s]
                       all        141        141      0.878       0.91      0.964       0.91
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        145/500      4.71G     0.4329     0.4067     0.9645          9        640: 100% 63/63 [00:16<00:00,  3.71it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.89it/s]
                       all        141        141      0.928      0.883      0.947      0.892
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        146/500      4.72G     0.4404     0.4165     0.9639         14        640: 100% 63/63 [00:17<00:00,  3.68it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.64it/s]
                       all        141        141      0.923      0.876      0.955      0.902
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        147/500      4.74G     0.4321        0.4     0.9616          7        640: 100% 63/63 [00:17<00:00,  3.54it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.87it/s]
                       all        141        141      0.935      0.873      0.962      0.906
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        148/500      4.76G     0.4471     0.4179     0.9684          7        640: 100% 63/63 [00:17<00:00,  3.70it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.53it/s]
                       all        141        141       0.96       0.91      0.963      0.903
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        149/500      4.77G     0.4268     0.4124     0.9543         12        640: 100% 63/63 [00:17<00:00,  3.52it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.16it/s]
                       all        141        141      0.934       0.89      0.961      0.915
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        150/500      4.78G     0.4295     0.3932     0.9613         10        640: 100% 63/63 [00:17<00:00,  3.70it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.73it/s]
                       all        141        141      0.925      0.903      0.954      0.899
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        151/500       4.8G     0.4363     0.4227     0.9635         11        640: 100% 63/63 [00:17<00:00,  3.52it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.90it/s]
                       all        141        141      0.874      0.895      0.963      0.905
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        152/500      4.82G     0.4108      0.379      0.945          8        640: 100% 63/63 [00:16<00:00,  3.75it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.21it/s]
                       all        141        141      0.933       0.87      0.961      0.906
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        153/500      4.83G     0.4324     0.4062     0.9626          8        640: 100% 63/63 [00:17<00:00,  3.65it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  3.31it/s]
                       all        141        141      0.927      0.902      0.963      0.911
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        154/500      4.84G     0.4343     0.4099     0.9669         12        640: 100% 63/63 [00:16<00:00,  3.75it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.61it/s]
                       all        141        141      0.918      0.896      0.932      0.881
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        155/500      4.86G     0.4241     0.3931      0.951          9        640: 100% 63/63 [00:16<00:00,  3.79it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.46it/s]
                       all        141        141       0.91       0.91      0.966      0.918
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        156/500      4.87G     0.4295     0.3876     0.9556          5        640: 100% 63/63 [00:17<00:00,  3.60it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.08it/s]
                       all        141        141       0.91      0.912      0.966      0.916
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        157/500      4.89G     0.4321     0.3948      0.968         11        640: 100% 63/63 [00:16<00:00,  3.78it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.33it/s]
                       all        141        141      0.915      0.911      0.954      0.898
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        158/500       4.9G      0.417     0.3855     0.9583          7        640: 100% 63/63 [00:17<00:00,  3.52it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.27it/s]
                       all        141        141      0.895      0.881      0.937      0.886
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        159/500      4.92G     0.4297     0.3977     0.9581          9        640: 100% 63/63 [00:16<00:00,  3.79it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.89it/s]
                       all        141        141      0.878      0.902      0.943      0.888
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        160/500      4.93G     0.4076     0.3648     0.9551         13        640: 100% 63/63 [00:16<00:00,  3.78it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  3.04it/s]
                       all        141        141       0.89      0.916      0.968      0.913
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        161/500      4.95G     0.4053     0.3788     0.9429         11        640: 100% 63/63 [00:16<00:00,  3.73it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.52it/s]
                       all        141        141      0.917      0.869      0.952      0.904
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        162/500      4.96G     0.4239     0.3591     0.9599         10        640: 100% 63/63 [00:16<00:00,  3.73it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.38it/s]
                       all        141        141      0.936      0.893      0.932      0.884
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        163/500      4.98G     0.4152     0.3711       0.95          7        640: 100% 63/63 [00:17<00:00,  3.59it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.76it/s]
                       all        141        141      0.938       0.87      0.949      0.898
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        164/500      4.99G     0.4271     0.4065     0.9587         13        640: 100% 63/63 [00:17<00:00,  3.68it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.17it/s]
                       all        141        141      0.898      0.882      0.931      0.886
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        165/500      5.01G     0.4038     0.3626     0.9407         11        640: 100% 63/63 [00:18<00:00,  3.47it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.13it/s]
                       all        141        141      0.911      0.893      0.957      0.908
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        166/500      5.02G     0.4187     0.3832     0.9473         11        640: 100% 63/63 [00:16<00:00,  3.77it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.70it/s]
                       all        141        141      0.861      0.889      0.951      0.899
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        167/500      5.04G     0.4071     0.3698     0.9475         11        640: 100% 63/63 [00:17<00:00,  3.65it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  3.50it/s]
                       all        141        141      0.911      0.886      0.951      0.891
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        168/500      5.05G      0.398     0.3567     0.9466          9        640: 100% 63/63 [00:16<00:00,  3.76it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.92it/s]
                       all        141        141      0.926      0.832      0.955      0.893
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        169/500      5.07G     0.4089     0.3749     0.9517         11        640: 100% 63/63 [00:16<00:00,  3.78it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  3.97it/s]
                       all        141        141      0.935      0.865      0.948      0.892
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        170/500      5.08G      0.401     0.3599      0.947         13        640: 100% 63/63 [00:17<00:00,  3.60it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.46it/s]
                       all        141        141      0.897      0.886      0.933      0.879
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        171/500       5.1G      0.405     0.3682       0.95          8        640: 100% 63/63 [00:16<00:00,  3.73it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.68it/s]
                       all        141        141      0.924      0.863      0.946      0.894
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        172/500      5.11G     0.4183     0.3729     0.9493          9        640: 100% 63/63 [00:17<00:00,  3.53it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.28it/s]
                       all        141        141      0.923      0.886      0.934      0.882
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        173/500      5.13G     0.4131      0.382     0.9537         10        640: 100% 63/63 [00:16<00:00,  3.75it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.28it/s]
                       all        141        141      0.898        0.9      0.957      0.902
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        174/500      5.13G     0.4079     0.3757     0.9393         12        640: 100% 63/63 [00:17<00:00,  3.55it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.78it/s]
                       all        141        141      0.889      0.882      0.932      0.887
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        175/500      5.16G      0.398     0.3658     0.9436         11        640: 100% 63/63 [00:16<00:00,  3.76it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.69it/s]
                       all        141        141      0.898      0.883      0.942      0.892
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        176/500      5.17G     0.3947     0.3564     0.9465          8        640: 100% 63/63 [00:17<00:00,  3.70it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  3.14it/s]
                       all        141        141      0.907      0.857      0.936      0.886
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        177/500      5.19G     0.4043     0.3645     0.9445          8        640: 100% 63/63 [00:16<00:00,  3.74it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.27it/s]
                       all        141        141      0.908      0.859      0.935      0.883
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        178/500      5.19G     0.3938     0.3548     0.9336          7        640: 100% 63/63 [00:16<00:00,  3.76it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.52it/s]
                       all        141        141      0.933      0.836       0.95      0.897
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        179/500      5.22G     0.3906     0.3556     0.9419          9        640: 100% 63/63 [00:17<00:00,  3.60it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.42it/s]
                       all        141        141      0.915      0.886      0.957      0.906
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        180/500      5.23G     0.3891     0.3466     0.9324          7        640: 100% 63/63 [00:16<00:00,  3.79it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.73it/s]
                       all        141        141      0.939       0.86      0.973      0.918
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        181/500      5.24G     0.4008     0.3665     0.9438          6        640: 100% 63/63 [00:17<00:00,  3.59it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.32it/s]
                       all        141        141      0.902      0.911      0.958      0.908
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        182/500      5.25G     0.3979     0.3422     0.9406         13        640: 100% 63/63 [00:16<00:00,  3.85it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.17it/s]
                       all        141        141      0.906      0.899       0.96       0.91
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        183/500      5.27G     0.4004     0.3586     0.9478          9        640: 100% 63/63 [00:16<00:00,  3.77it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  3.78it/s]
                       all        141        141       0.91      0.871       0.95      0.898
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        184/500      5.29G     0.3936     0.3533     0.9315          7        640: 100% 63/63 [00:17<00:00,  3.68it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.67it/s]
                       all        141        141      0.845      0.912      0.953      0.903
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        185/500       5.3G     0.3922     0.3382     0.9367         10        640: 100% 63/63 [00:16<00:00,  3.77it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.91it/s]
                       all        141        141       0.93      0.896       0.96      0.906
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        186/500      5.31G     0.4038     0.3527     0.9531          8        640: 100% 63/63 [00:17<00:00,  3.52it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.88it/s]
                       all        141        141      0.938      0.885      0.975      0.928
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        187/500      5.33G     0.4033     0.3489     0.9435          8        640: 100% 63/63 [00:16<00:00,  3.73it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.68it/s]
                       all        141        141      0.925      0.886      0.976       0.92
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        188/500      5.35G     0.4045     0.3627     0.9506         11        640: 100% 63/63 [00:17<00:00,  3.53it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.38it/s]
                       all        141        141      0.916      0.909      0.964      0.905
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        189/500      5.37G     0.4015     0.3718     0.9523         10        640: 100% 63/63 [00:16<00:00,  3.76it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.55it/s]
                       all        141        141       0.93      0.861      0.957        0.9
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        190/500      5.37G     0.3902     0.3508     0.9382         12        640: 100% 63/63 [00:17<00:00,  3.67it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  3.09it/s]
                       all        141        141      0.938        0.9      0.975      0.923
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        191/500      5.39G     0.4023     0.3504     0.9415         13        640: 100% 63/63 [00:16<00:00,  3.71it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.46it/s]
                       all        141        141      0.899      0.924      0.962      0.907
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        192/500      5.41G     0.4112     0.3542     0.9529          6        640: 100% 63/63 [00:16<00:00,  3.75it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.41it/s]
                       all        141        141      0.881      0.896      0.965      0.912
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        193/500      5.42G     0.3867     0.3386     0.9413         10        640: 100% 63/63 [00:17<00:00,  3.57it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.52it/s]
                       all        141        141      0.893      0.923      0.959      0.912
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        194/500      5.43G     0.3875     0.3363     0.9364          6        640: 100% 63/63 [00:16<00:00,  3.78it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.12it/s]
                       all        141        141      0.944      0.865      0.959      0.905
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        195/500      5.45G     0.3928     0.3395     0.9373          8        640: 100% 63/63 [00:17<00:00,  3.51it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.19it/s]
                       all        141        141      0.937      0.898      0.964      0.915
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        196/500      5.47G     0.3867     0.3412     0.9398         12        640: 100% 63/63 [00:16<00:00,  3.75it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.23it/s]
                       all        141        141      0.951      0.902      0.973      0.917
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        197/500      5.48G       0.39     0.3383     0.9359         11        640: 100% 63/63 [00:17<00:00,  3.57it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.08it/s]
                       all        141        141      0.932      0.894      0.956      0.897
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        198/500      5.49G     0.3955      0.353     0.9429         10        640: 100% 63/63 [00:16<00:00,  3.75it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.90it/s]
                       all        141        141       0.91      0.905      0.954       0.89
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        199/500      5.51G      0.381     0.3333     0.9417          7        640: 100% 63/63 [00:16<00:00,  3.74it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  3.88it/s]
                       all        141        141      0.904      0.932       0.97      0.916
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        200/500      5.53G     0.3846     0.3373     0.9305         11        640: 100% 63/63 [00:17<00:00,  3.64it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.48it/s]
                       all        141        141      0.926      0.901      0.965      0.909
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        201/500      5.54G     0.3981     0.3455     0.9442          7        640: 100% 63/63 [00:17<00:00,  3.63it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.54it/s]
                       all        141        141      0.924      0.914      0.976      0.921
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        202/500      5.55G     0.3918     0.3332     0.9427          7        640: 100% 63/63 [00:18<00:00,  3.49it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.01it/s]
                       all        141        141      0.918      0.873      0.978      0.918
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        203/500      5.57G     0.4005     0.3505     0.9417          5        640: 100% 63/63 [00:17<00:00,  3.67it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.62it/s]
                       all        141        141      0.915      0.884      0.962       0.91
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        204/500      5.59G     0.3939     0.3528     0.9457          9        640: 100% 63/63 [00:18<00:00,  3.44it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  5.00it/s]
                       all        141        141       0.95      0.904      0.971      0.921
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        205/500       5.6G     0.3951     0.3331     0.9393         10        640: 100% 63/63 [00:17<00:00,  3.67it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.28it/s]
                       all        141        141      0.937      0.899      0.961      0.909
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        206/500      5.61G     0.3779     0.3195     0.9257          4        640: 100% 63/63 [00:17<00:00,  3.52it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.40it/s]
                       all        141        141      0.902      0.904      0.958      0.909
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        207/500      5.63G      0.388     0.3289      0.937          8        640: 100% 63/63 [00:17<00:00,  3.70it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.51it/s]
                       all        141        141      0.931      0.918      0.969      0.911
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        208/500      5.64G     0.3792     0.3322     0.9343          7        640: 100% 63/63 [00:17<00:00,  3.65it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  2.97it/s]
                       all        141        141      0.929      0.911      0.959      0.905
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        209/500      5.66G     0.3787     0.3286     0.9327          6        640: 100% 63/63 [00:17<00:00,  3.69it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.29it/s]
                       all        141        141       0.94      0.902      0.962      0.906
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        210/500      5.67G     0.3671     0.3183     0.9236          5        640: 100% 63/63 [00:17<00:00,  3.66it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  3.49it/s]
                       all        141        141      0.929      0.871      0.961      0.907
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        211/500      5.69G     0.3717     0.3166     0.9253         14        640: 100% 63/63 [00:17<00:00,  3.57it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.59it/s]
                       all        141        141        0.9      0.885      0.968      0.912
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        212/500      5.71G     0.3731     0.3157     0.9365          7        640: 100% 63/63 [00:17<00:00,  3.68it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.44it/s]
                       all        141        141      0.909      0.884      0.955      0.902
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        213/500      5.72G     0.3845     0.3225     0.9456          9        640: 100% 63/63 [00:18<00:00,  3.49it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.47it/s]
                       all        141        141      0.909      0.907       0.97      0.922
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        214/500      5.73G     0.3777     0.3144     0.9351          8        640: 100% 63/63 [00:17<00:00,  3.69it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.61it/s]
                       all        141        141      0.935      0.889      0.961      0.914
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        215/500      5.75G     0.3684     0.3218     0.9249          8        640: 100% 63/63 [00:18<00:00,  3.46it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.80it/s]
                       all        141        141      0.896      0.903      0.965      0.916
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        216/500      5.76G      0.377     0.3212     0.9322         10        640: 100% 63/63 [00:16<00:00,  3.73it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.75it/s]
                       all        141        141      0.935      0.904      0.966      0.917
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        217/500      5.78G     0.3798     0.3433      0.942          9        640: 100% 63/63 [00:18<00:00,  3.43it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.49it/s]
                       all        141        141      0.903      0.914      0.958      0.913
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        218/500      5.79G     0.3771     0.3289     0.9425          9        640: 100% 63/63 [00:17<00:00,  3.69it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.70it/s]
                       all        141        141      0.935      0.893      0.968      0.911
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        219/500      5.81G     0.3636     0.3228     0.9312         10        640: 100% 63/63 [00:17<00:00,  3.64it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  3.59it/s]
                       all        141        141      0.908      0.908      0.965       0.91
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        220/500      5.82G     0.3753     0.3227     0.9282         10        640: 100% 63/63 [00:17<00:00,  3.68it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.10it/s]
                       all        141        141      0.946      0.874      0.966      0.907
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        221/500      5.84G     0.3742     0.3212     0.9398         11        640: 100% 63/63 [00:16<00:00,  3.72it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.39it/s]
                       all        141        141      0.944      0.896      0.971      0.916
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        222/500      5.84G     0.3802      0.333     0.9399          7        640: 100% 63/63 [00:17<00:00,  3.60it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.56it/s]
                       all        141        141      0.921      0.912      0.971      0.922
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        223/500      5.87G     0.3863     0.3265     0.9346          7        640: 100% 63/63 [00:17<00:00,  3.68it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.64it/s]
                       all        141        141       0.95      0.896      0.976      0.925
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        224/500      5.88G      0.357     0.3115     0.9239          9        640: 100% 63/63 [00:17<00:00,  3.52it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.93it/s]
                       all        141        141      0.945      0.879      0.973      0.929
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        225/500       5.9G     0.3754     0.3266     0.9355         10        640: 100% 63/63 [00:16<00:00,  3.71it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.71it/s]
                       all        141        141      0.919       0.91       0.97      0.919
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        226/500       5.9G     0.3645      0.308     0.9286          9        640: 100% 63/63 [00:17<00:00,  3.55it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.01it/s]
                       all        141        141      0.919       0.91       0.96      0.904
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        227/500      5.93G      0.368     0.3203     0.9327          5        640: 100% 63/63 [00:16<00:00,  3.79it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.02it/s]
                       all        141        141      0.919      0.889      0.958      0.906
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        228/500      5.94G     0.3713     0.3225     0.9308          9        640: 100% 63/63 [00:17<00:00,  3.61it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  3.19it/s]
                       all        141        141       0.92      0.873      0.963       0.91
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        229/500      5.96G      0.374     0.3253     0.9353         11        640: 100% 63/63 [00:17<00:00,  3.68it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.50it/s]
                       all        141        141      0.883      0.916      0.971      0.915
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        230/500      5.96G     0.3766     0.3199     0.9388          8        640: 100% 63/63 [00:17<00:00,  3.69it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.73it/s]
                       all        141        141      0.909      0.902      0.962      0.912
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        231/500      5.99G     0.3661     0.3204     0.9272          8        640: 100% 63/63 [00:17<00:00,  3.55it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.59it/s]
                       all        141        141        0.9      0.901      0.962      0.911
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        232/500         6G     0.3679     0.3178     0.9271         13        640: 100% 63/63 [00:16<00:00,  3.74it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.00it/s]
                       all        141        141      0.892      0.908      0.968      0.912
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        233/500      6.02G     0.3777     0.3094     0.9393          5        640: 100% 63/63 [00:18<00:00,  3.47it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.85it/s]
                       all        141        141      0.928      0.884      0.973      0.919
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        234/500      6.02G     0.3578     0.3056     0.9289         11        640: 100% 63/63 [00:17<00:00,  3.62it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.48it/s]
                       all        141        141       0.96      0.885      0.974      0.916
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        235/500      6.05G     0.3733     0.3364     0.9324          9        640: 100% 63/63 [00:18<00:00,  3.48it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.58it/s]
                       all        141        141      0.922      0.897      0.962      0.905
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        236/500      6.06G     0.3744     0.3197     0.9342          7        640: 100% 63/63 [00:17<00:00,  3.64it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.08it/s]
                       all        141        141      0.934      0.923      0.969      0.916
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        237/500      6.08G     0.3692     0.3124     0.9382         10        640: 100% 63/63 [00:17<00:00,  3.55it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  3.63it/s]
                       all        141        141      0.932      0.891      0.961      0.907
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        238/500      6.08G     0.3725     0.3197     0.9369          9        640: 100% 63/63 [00:16<00:00,  3.76it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.34it/s]
                       all        141        141      0.907      0.894      0.953      0.899
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        239/500      6.11G      0.358     0.3037     0.9242         13        640: 100% 63/63 [00:17<00:00,  3.69it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  2.74it/s]
                       all        141        141      0.892      0.873      0.951      0.899
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        240/500      6.12G     0.3506     0.2882     0.9233          8        640: 100% 63/63 [00:16<00:00,  3.79it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.59it/s]
                       all        141        141      0.878       0.89      0.948      0.896
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        241/500      6.13G     0.3688     0.3075     0.9273          8        640: 100% 63/63 [00:16<00:00,  3.76it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.46it/s]
                       all        141        141       0.91      0.877      0.947       0.89
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        242/500      6.14G     0.3681     0.3075     0.9283          8        640: 100% 63/63 [00:17<00:00,  3.50it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.22it/s]
                       all        141        141      0.913      0.877      0.956      0.898
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        243/500      6.16G     0.3704     0.3105     0.9264         12        640: 100% 63/63 [00:16<00:00,  3.73it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.75it/s]
                       all        141        141      0.907      0.907      0.964      0.911
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        244/500      6.18G     0.3569     0.3068     0.9213         15        640: 100% 63/63 [00:17<00:00,  3.51it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.61it/s]
                       all        141        141      0.943      0.893       0.96      0.903
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        245/500      6.19G     0.3636     0.3094     0.9273          9        640: 100% 63/63 [00:16<00:00,  3.74it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.54it/s]
                       all        141        141      0.932       0.87      0.954      0.903
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        246/500       6.2G     0.3666      0.326     0.9235         10        640: 100% 63/63 [00:16<00:00,  3.77it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  2.97it/s]
                       all        141        141      0.929      0.889      0.958      0.899
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        247/500      6.22G     0.3618     0.3067     0.9269          7        640: 100% 63/63 [00:16<00:00,  3.74it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.73it/s]
                       all        141        141      0.877      0.902      0.957      0.893
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        248/500      6.24G     0.3435     0.3034     0.9182         11        640: 100% 63/63 [00:16<00:00,  3.80it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.69it/s]
                       all        141        141      0.884      0.889      0.948      0.885
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        249/500      6.25G     0.3552      0.285     0.9267         13        640: 100% 63/63 [00:17<00:00,  3.51it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.13it/s]
                       all        141        141      0.928      0.884      0.952      0.896
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        250/500      6.26G     0.3454     0.2928     0.9169          8        640: 100% 63/63 [00:16<00:00,  3.77it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.49it/s]
                       all        141        141      0.938      0.863      0.952      0.893
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        251/500      6.28G     0.3558     0.2974     0.9292          8        640: 100% 63/63 [00:17<00:00,  3.57it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.66it/s]
                       all        141        141      0.942      0.875      0.966      0.914
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        252/500       6.3G     0.3494     0.2877     0.9192         11        640: 100% 63/63 [00:16<00:00,  3.79it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  5.00it/s]
                       all        141        141      0.933      0.897      0.966      0.915
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        253/500      6.31G     0.3545     0.2981     0.9257          8        640: 100% 63/63 [00:16<00:00,  3.79it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  3.82it/s]
                       all        141        141      0.885      0.898      0.955      0.903
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        254/500      6.32G     0.3521     0.2854     0.9235          7        640: 100% 63/63 [00:17<00:00,  3.66it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.94it/s]
                       all        141        141      0.935      0.867      0.964      0.912
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        255/500      6.34G     0.3573     0.2983      0.933         10        640: 100% 63/63 [00:16<00:00,  3.75it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.78it/s]
                       all        141        141      0.938      0.883      0.953      0.904
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        256/500      6.36G     0.3476      0.301     0.9232          6        640: 100% 63/63 [00:18<00:00,  3.49it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.53it/s]
                       all        141        141      0.924      0.871      0.955      0.908
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        257/500      6.37G     0.3328     0.2717     0.9115          5        640: 100% 63/63 [00:16<00:00,  3.75it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.13it/s]
                       all        141        141      0.916      0.871      0.955      0.903
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        258/500      6.38G     0.3545     0.2945     0.9254         10        640: 100% 63/63 [00:17<00:00,  3.54it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.74it/s]
                       all        141        141      0.912        0.9      0.972      0.916
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        259/500       6.4G     0.3497     0.2939     0.9225          7        640: 100% 63/63 [00:16<00:00,  3.76it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.82it/s]
                       all        141        141      0.907      0.905      0.972      0.921
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        260/500      6.42G     0.3453     0.2865      0.921          8        640: 100% 63/63 [00:16<00:00,  3.73it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  3.10it/s]
                       all        141        141      0.931      0.856      0.965      0.911
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        261/500      6.43G     0.3382     0.2816     0.9188          8        640: 100% 63/63 [00:16<00:00,  3.76it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.34it/s]
                       all        141        141      0.932      0.834      0.967      0.912
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        262/500      6.44G     0.3417     0.2933     0.9217         10        640: 100% 63/63 [00:16<00:00,  3.76it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.94it/s]
                       all        141        141      0.896      0.883       0.96      0.904
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        263/500      6.46G     0.3458     0.2839     0.9276          8        640: 100% 63/63 [00:17<00:00,  3.54it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.97it/s]
                       all        141        141      0.929      0.897      0.961      0.906
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        264/500      6.47G     0.3359     0.2665     0.9168          8        640: 100% 63/63 [00:16<00:00,  3.78it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.19it/s]
                       all        141        141      0.922      0.892      0.961      0.908
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        265/500      6.49G     0.3415     0.2785     0.9193          9        640: 100% 63/63 [00:17<00:00,  3.55it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.72it/s]
                       all        141        141      0.923      0.902      0.973      0.919
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        266/500       6.5G     0.3403     0.2878     0.9215          8        640: 100% 63/63 [00:16<00:00,  3.74it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.33it/s]
                       all        141        141      0.908       0.92      0.976      0.921
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        267/500      6.52G     0.3372     0.2789     0.9197          7        640: 100% 63/63 [00:17<00:00,  3.62it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  3.58it/s]
                       all        141        141      0.926      0.908      0.969      0.917
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        268/500      6.54G     0.3423     0.2867     0.9128          9        640: 100% 63/63 [00:16<00:00,  3.82it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  3.90it/s]
                       all        141        141      0.935      0.891       0.97      0.919
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        269/500      6.55G     0.3438      0.295     0.9284          7        640: 100% 63/63 [00:16<00:00,  3.77it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.64it/s]
                       all        141        141      0.937      0.898      0.972      0.919
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        270/500      6.56G     0.3336     0.2807     0.9146          7        640: 100% 63/63 [00:17<00:00,  3.56it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.05it/s]
                       all        141        141      0.908      0.912      0.974      0.919
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        271/500      6.58G     0.3326     0.2779     0.9158          8        640: 100% 63/63 [00:16<00:00,  3.81it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.08it/s]
                       all        141        141      0.924       0.88      0.966      0.916
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        272/500      6.59G     0.3432     0.2829     0.9208          6        640: 100% 63/63 [00:17<00:00,  3.55it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.54it/s]
                       all        141        141      0.902      0.895      0.964      0.913
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        273/500      6.61G     0.3511     0.2807      0.923         14        640: 100% 63/63 [00:17<00:00,  3.70it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.40it/s]
                       all        141        141      0.863      0.941      0.966      0.912
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        274/500      6.62G     0.3494     0.2916     0.9266         13        640: 100% 63/63 [00:17<00:00,  3.61it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  3.45it/s]
                       all        141        141      0.927      0.898      0.966      0.916
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        275/500      6.64G     0.3301     0.2758     0.9179          8        640: 100% 63/63 [00:16<00:00,  3.75it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.38it/s]
                       all        141        141      0.933      0.901      0.975      0.921
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        276/500      6.65G     0.3368     0.2765     0.9125          8        640: 100% 63/63 [00:16<00:00,  3.75it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.12it/s]
                       all        141        141      0.944      0.891      0.984      0.934
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        277/500      6.67G     0.3356     0.2837     0.9155         12        640: 100% 63/63 [00:17<00:00,  3.56it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.32it/s]
                       all        141        141      0.914      0.889      0.972      0.914
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        278/500      6.68G     0.3363     0.2813     0.9166          9        640: 100% 63/63 [00:17<00:00,  3.68it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.36it/s]
                       all        141        141      0.923      0.917      0.977      0.922
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        279/500       6.7G     0.3354      0.277     0.9165          9        640: 100% 63/63 [00:17<00:00,  3.54it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.17it/s]
                       all        141        141      0.921      0.931      0.979      0.922
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        280/500      6.71G     0.3449     0.2835     0.9203         11        640: 100% 63/63 [00:16<00:00,  3.79it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.12it/s]
                       all        141        141      0.927      0.935      0.979      0.921
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        281/500      6.73G     0.3435     0.2861     0.9209         10        640: 100% 63/63 [00:17<00:00,  3.60it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  3.07it/s]
                       all        141        141      0.911      0.932      0.977      0.922
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        282/500      6.73G     0.3438     0.2994     0.9277         12        640: 100% 63/63 [00:16<00:00,  3.75it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.78it/s]
                       all        141        141      0.909      0.908       0.97      0.916
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        283/500      6.76G     0.3405     0.2789     0.9224         11        640: 100% 63/63 [00:16<00:00,  3.71it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.29it/s]
                       all        141        141      0.926      0.899      0.968      0.916
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        284/500      6.77G     0.3337      0.268     0.9135          8        640: 100% 63/63 [00:17<00:00,  3.61it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.48it/s]
                       all        141        141      0.912      0.901      0.972      0.922
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        285/500      6.79G     0.3551     0.2927     0.9277          9        640: 100% 63/63 [00:16<00:00,  3.77it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.39it/s]
                       all        141        141       0.92      0.882      0.966      0.916
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        286/500      6.79G     0.3458     0.2825     0.9246          7        640: 100% 63/63 [00:17<00:00,  3.51it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.09it/s]
                       all        141        141      0.929       0.87      0.972      0.921
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        287/500      6.82G     0.3506     0.2942     0.9229          9        640: 100% 63/63 [00:16<00:00,  3.73it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.64it/s]
                       all        141        141      0.913      0.894      0.956      0.906
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        288/500      6.83G     0.3328     0.2783     0.9156          6        640: 100% 63/63 [00:17<00:00,  3.66it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  2.88it/s]
                       all        141        141       0.93      0.896      0.964      0.909
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        289/500      6.85G     0.3333     0.2768     0.9148         10        640: 100% 63/63 [00:17<00:00,  3.69it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.64it/s]
                       all        141        141      0.927      0.901      0.959      0.904
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        290/500      6.85G     0.3322     0.2761     0.9241         12        640: 100% 63/63 [00:17<00:00,  3.69it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  3.86it/s]
                       all        141        141      0.915      0.904      0.954      0.898
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        291/500      6.88G      0.334     0.2791     0.9232         11        640: 100% 63/63 [00:17<00:00,  3.66it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.46it/s]
                       all        141        141      0.935      0.893      0.954      0.896
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        292/500      6.89G     0.3312     0.2653     0.9154         10        640: 100% 63/63 [00:16<00:00,  3.76it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.41it/s]
                       all        141        141      0.929      0.872      0.959      0.903
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        293/500      6.91G     0.3281     0.2738     0.9109         12        640: 100% 63/63 [00:17<00:00,  3.51it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  6.07it/s]
                       all        141        141      0.889      0.909      0.963       0.91
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        294/500      6.91G     0.3286     0.2697     0.9112         11        640: 100% 63/63 [00:16<00:00,  3.71it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.60it/s]
                       all        141        141      0.917      0.892      0.959      0.902
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        295/500      6.94G     0.3361     0.2859     0.9175         11        640: 100% 63/63 [00:17<00:00,  3.59it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  3.44it/s]
                       all        141        141      0.934      0.893      0.947      0.898
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        296/500      6.95G     0.3249     0.2743     0.9157          9        640: 100% 63/63 [00:16<00:00,  3.71it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.49it/s]
                       all        141        141      0.934      0.866      0.943      0.897
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        297/500      6.96G     0.3248     0.2705      0.911         12        640: 100% 63/63 [00:16<00:00,  3.74it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.01it/s]
                       all        141        141      0.924       0.86      0.955      0.907
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        298/500      6.97G     0.3178     0.2572     0.9077          8        640: 100% 63/63 [00:17<00:00,  3.55it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.50it/s]
                       all        141        141      0.922      0.882      0.961      0.908
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        299/500      6.99G      0.321     0.2645      0.908         11        640: 100% 63/63 [00:17<00:00,  3.67it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.54it/s]
                       all        141        141        0.9      0.902      0.971      0.917
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        300/500      7.01G     0.3188     0.2621     0.9074         12        640: 100% 63/63 [00:17<00:00,  3.53it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.92it/s]
                       all        141        141       0.89      0.888      0.967      0.914
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        301/500      7.02G     0.3181     0.2613     0.9098         13        640: 100% 63/63 [00:17<00:00,  3.68it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.48it/s]
                       all        141        141      0.899      0.905      0.966      0.913
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        302/500      7.03G     0.3356     0.2728      0.914          9        640: 100% 63/63 [00:17<00:00,  3.56it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.17it/s]
                       all        141        141      0.903      0.923       0.97      0.921
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        303/500      7.05G      0.324     0.2652     0.9165          9        640: 100% 63/63 [00:17<00:00,  3.65it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.43it/s]
                       all        141        141      0.934      0.883      0.963      0.913
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        304/500      7.07G     0.3189      0.265     0.9123          7        640: 100% 63/63 [00:17<00:00,  3.67it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  3.11it/s]
                       all        141        141      0.937      0.868      0.961      0.908
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        305/500      7.08G     0.3227     0.2625     0.9122         16        640: 100% 63/63 [00:16<00:00,  3.73it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.93it/s]
                       all        141        141      0.924      0.904      0.979       0.92
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        306/500      7.09G     0.3302     0.2781     0.9117         10        640: 100% 63/63 [00:16<00:00,  3.71it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.75it/s]
                       all        141        141      0.943      0.881      0.976      0.919
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        307/500      7.11G     0.3205     0.2612     0.9081         10        640: 100% 63/63 [00:17<00:00,  3.57it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.25it/s]
                       all        141        141      0.917      0.883      0.971      0.918
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        308/500      7.13G     0.3196     0.2545     0.9125          7        640: 100% 63/63 [00:16<00:00,  3.74it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.79it/s]
                       all        141        141      0.953      0.898      0.974      0.921
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        309/500      7.14G     0.3208     0.2583     0.9047          9        640: 100% 63/63 [00:17<00:00,  3.56it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.87it/s]
                       all        141        141      0.937      0.906      0.979      0.928
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        310/500      7.15G     0.3196     0.2587     0.9127         13        640: 100% 63/63 [00:16<00:00,  3.75it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.42it/s]
                       all        141        141      0.941      0.904      0.977      0.922
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        311/500      7.17G     0.3121     0.2614     0.9026          9        640: 100% 63/63 [00:16<00:00,  3.74it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  3.33it/s]
                       all        141        141      0.934      0.894      0.964      0.916
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        312/500      7.19G     0.3335     0.2739     0.9163         11        640: 100% 63/63 [00:16<00:00,  3.74it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.47it/s]
                       all        141        141      0.897      0.911      0.961      0.909
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        313/500       7.2G     0.3123     0.2602     0.8995         11        640: 100% 63/63 [00:16<00:00,  3.75it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.26it/s]
                       all        141        141      0.902      0.908      0.956      0.905
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        314/500      7.21G     0.3174     0.2531     0.9062         10        640: 100% 63/63 [00:17<00:00,  3.53it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.34it/s]
                       all        141        141      0.915        0.9      0.965      0.914
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        315/500      7.23G     0.3295     0.2706     0.9214          9        640: 100% 63/63 [00:16<00:00,  3.74it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.69it/s]
                       all        141        141      0.925      0.899      0.961      0.912
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        316/500      7.25G     0.3235     0.2649     0.9114          8        640: 100% 63/63 [00:17<00:00,  3.57it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  3.76it/s]
                       all        141        141      0.933      0.881       0.96      0.911
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        317/500      7.26G     0.3141     0.2553     0.9035         14        640: 100% 63/63 [00:16<00:00,  3.75it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.77it/s]
                       all        141        141      0.875      0.925      0.961       0.91
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        318/500      7.27G     0.3227     0.2635     0.9051         10        640: 100% 63/63 [00:17<00:00,  3.70it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  3.22it/s]
                       all        141        141      0.947       0.88      0.969      0.914
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        319/500      7.29G     0.3123     0.2471     0.9061          6        640: 100% 63/63 [00:17<00:00,  3.69it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.85it/s]
                       all        141        141      0.944      0.905       0.97      0.912
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        320/500       7.3G     0.3115     0.2632     0.9021         10        640: 100% 63/63 [00:16<00:00,  3.78it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.77it/s]
                       all        141        141      0.902      0.898      0.961      0.908
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        321/500      7.32G     0.3116     0.2449     0.9067         12        640: 100% 63/63 [00:18<00:00,  3.48it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.75it/s]
                       all        141        141      0.916       0.88       0.96      0.908
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        322/500      7.33G     0.3115     0.2536     0.9039          7        640: 100% 63/63 [00:17<00:00,  3.66it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.36it/s]
                       all        141        141      0.915      0.884      0.963      0.911
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        323/500      7.35G     0.3074      0.246     0.8978         10        640: 100% 63/63 [00:17<00:00,  3.53it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.00it/s]
                       all        141        141      0.914      0.896      0.969      0.918
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        324/500      7.37G      0.304     0.2423     0.9031          8        640: 100% 63/63 [00:16<00:00,  3.74it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.45it/s]
                       all        141        141      0.905      0.913      0.972      0.919
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        325/500      7.38G     0.3064     0.2487     0.9061          8        640: 100% 63/63 [00:17<00:00,  3.62it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.92it/s]
                       all        141        141      0.923      0.876      0.963       0.91
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        326/500      2.51G     0.3117     0.2575     0.9004         13        640: 100% 63/63 [00:17<00:00,  3.66it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.10it/s]
                       all        141        141      0.929      0.886       0.97      0.921
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        327/500      2.51G     0.3189     0.2674     0.9105         12        640: 100% 63/63 [00:17<00:00,  3.51it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.72it/s]
                       all        141        141      0.936      0.897      0.973      0.923
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        328/500      2.51G     0.3204     0.2565     0.9099         11        640: 100% 63/63 [00:16<00:00,  3.76it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.50it/s]
                       all        141        141      0.932      0.909       0.97      0.919
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        329/500      2.51G     0.3144     0.2507     0.9078          8        640: 100% 63/63 [00:17<00:00,  3.64it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  3.06it/s]
                       all        141        141       0.93      0.891      0.966      0.914
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        330/500      2.51G     0.3119     0.2531     0.9125          5        640: 100% 63/63 [00:17<00:00,  3.64it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.08it/s]
                       all        141        141       0.91      0.902      0.968      0.915
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        331/500      2.51G     0.3176     0.2589     0.9122         11        640: 100% 63/63 [00:17<00:00,  3.68it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  3.81it/s]
                       all        141        141      0.928      0.887      0.973      0.919
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        332/500      2.51G      0.317     0.2575     0.9111         11        640: 100% 63/63 [00:17<00:00,  3.59it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.27it/s]
                       all        141        141      0.919      0.906      0.971      0.923
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        333/500      2.51G     0.3062     0.2534     0.9102         13        640: 100% 63/63 [00:16<00:00,  3.74it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.48it/s]
                       all        141        141      0.926      0.899      0.963      0.912
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        334/500      2.51G     0.3031     0.2462     0.9004         10        640: 100% 63/63 [00:17<00:00,  3.51it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.36it/s]
                       all        141        141      0.931      0.892      0.959      0.911
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        335/500      2.51G     0.3102     0.2503     0.9106         10        640: 100% 63/63 [00:16<00:00,  3.72it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.41it/s]
                       all        141        141      0.919      0.898      0.963      0.911
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        336/500      2.51G     0.3052     0.2477     0.9019         12        640: 100% 63/63 [00:17<00:00,  3.53it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.41it/s]
                       all        141        141      0.925      0.892      0.964      0.911
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        337/500      2.51G     0.3062     0.2432      0.905         12        640: 100% 63/63 [00:17<00:00,  3.69it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.88it/s]
                       all        141        141      0.911      0.892      0.957      0.906
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        338/500      2.51G     0.3066     0.2491     0.9055         10        640: 100% 63/63 [00:17<00:00,  3.50it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  3.64it/s]
                       all        141        141      0.934      0.886      0.953      0.902
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        339/500      2.51G     0.3041     0.2491     0.9061          8        640: 100% 63/63 [00:17<00:00,  3.65it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.36it/s]
                       all        141        141      0.913      0.896      0.969      0.923
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        340/500      2.52G     0.3064     0.2465     0.9013         12        640: 100% 63/63 [00:17<00:00,  3.68it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  3.63it/s]
                       all        141        141      0.919      0.885      0.969      0.918
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        341/500      2.53G     0.3114     0.2511     0.9049          8        640: 100% 63/63 [00:17<00:00,  3.64it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.43it/s]
                       all        141        141       0.93      0.881      0.964      0.915
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        342/500      2.54G     0.3097     0.2605     0.9102         10        640: 100% 63/63 [00:17<00:00,  3.69it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.88it/s]
                       all        141        141      0.945      0.869      0.966       0.92
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        343/500      2.56G     0.3129     0.2573      0.904          8        640: 100% 63/63 [00:18<00:00,  3.45it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  3.40it/s]
                       all        141        141      0.938      0.887      0.962      0.911
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        344/500      2.58G     0.3061     0.2469     0.9045          9        640: 100% 63/63 [00:17<00:00,  3.67it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.25it/s]
                       all        141        141      0.925      0.895       0.96      0.909
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        345/500      2.59G     0.3064     0.2404     0.8982         11        640: 100% 63/63 [00:18<00:00,  3.48it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.82it/s]
                       all        141        141      0.901      0.904      0.959      0.911
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        346/500       2.6G     0.3014     0.2484     0.9026          8        640: 100% 63/63 [00:17<00:00,  3.70it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.19it/s]
                       all        141        141      0.924      0.876       0.96      0.914
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        347/500      2.62G     0.3044     0.2399     0.9022         13        640: 100% 63/63 [00:18<00:00,  3.50it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.81it/s]
                       all        141        141      0.906      0.898       0.96      0.911
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        348/500      2.63G     0.3015     0.2392     0.9048          9        640: 100% 63/63 [00:16<00:00,  3.72it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.75it/s]
                       all        141        141      0.875      0.924      0.968      0.919
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        349/500      2.65G     0.3018     0.2467      0.902         11        640: 100% 63/63 [00:17<00:00,  3.60it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  3.08it/s]
                       all        141        141      0.921       0.88      0.968      0.921
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        350/500      2.66G     0.3032     0.2467      0.903          8        640: 100% 63/63 [00:16<00:00,  3.73it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.97it/s]
                       all        141        141       0.91      0.899      0.967      0.922
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        351/500      2.68G     0.3016     0.2527     0.8983         10        640: 100% 63/63 [00:17<00:00,  3.69it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.10it/s]
                       all        141        141      0.934      0.882      0.966       0.92
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        352/500      2.69G     0.2936     0.2374     0.9001         12        640: 100% 63/63 [00:17<00:00,  3.54it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.94it/s]
                       all        141        141      0.939       0.89      0.968      0.921
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        353/500      2.71G     0.3078     0.2564     0.9074          8        640: 100% 63/63 [00:17<00:00,  3.71it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.75it/s]
                       all        141        141       0.93      0.912      0.969       0.92
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        354/500      2.71G     0.2936     0.2396     0.8991          5        640: 100% 63/63 [00:18<00:00,  3.47it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.97it/s]
                       all        141        141      0.934      0.895      0.976      0.927
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        355/500      2.74G     0.2862     0.2328      0.895         13        640: 100% 63/63 [00:16<00:00,  3.74it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.74it/s]
                       all        141        141      0.948      0.897      0.987      0.939
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        356/500      2.75G      0.291     0.2312     0.8941         13        640: 100% 63/63 [00:17<00:00,  3.65it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  3.12it/s]
                       all        141        141      0.928      0.906      0.968      0.913
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        357/500      2.77G     0.3044     0.2474     0.9044          8        640: 100% 63/63 [00:17<00:00,  3.67it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.71it/s]
                       all        141        141      0.945      0.873      0.968      0.915
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        358/500      2.78G     0.3009     0.2402     0.9009         10        640: 100% 63/63 [00:16<00:00,  3.71it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.76it/s]
                       all        141        141       0.93      0.885      0.965      0.915
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        359/500       2.8G     0.3043      0.253     0.9061         12        640: 100% 63/63 [00:18<00:00,  3.50it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.24it/s]
                       all        141        141      0.925      0.903      0.974      0.923
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        360/500      2.81G     0.3054     0.2441     0.9007         15        640: 100% 63/63 [00:17<00:00,  3.68it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.24it/s]
                       all        141        141       0.94      0.901      0.975      0.923
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        361/500      2.83G      0.299     0.2484     0.9012          8        640: 100% 63/63 [00:18<00:00,  3.43it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.52it/s]
                       all        141        141      0.954      0.902      0.978      0.923
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        362/500      2.83G     0.3002     0.2407     0.8996         11        640: 100% 63/63 [00:17<00:00,  3.60it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  2.95it/s]
                       all        141        141      0.952      0.902      0.974      0.922
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        363/500      2.86G     0.2962     0.2322     0.9038         13        640: 100% 63/63 [00:20<00:00,  3.14it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.67it/s]
                       all        141        141      0.955      0.896      0.972      0.921
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        364/500      2.87G     0.2858     0.2362        0.9         12        640: 100% 63/63 [00:17<00:00,  3.62it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.12it/s]
                       all        141        141      0.935      0.898       0.97      0.921
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        365/500      2.89G     0.2921     0.2361     0.8991         10        640: 100% 63/63 [00:18<00:00,  3.50it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.36it/s]
                       all        141        141      0.935      0.893      0.968      0.919
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        366/500      2.89G     0.2869      0.222     0.8979         12        640: 100% 63/63 [00:17<00:00,  3.62it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.41it/s]
                       all        141        141      0.926      0.893      0.959      0.907
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        367/500      2.92G     0.2958     0.2379     0.9026          6        640: 100% 63/63 [00:19<00:00,  3.30it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.23it/s]
                       all        141        141      0.928      0.894      0.962      0.912
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        368/500      2.93G     0.2948     0.2368      0.896         11        640: 100% 63/63 [00:17<00:00,  3.65it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.62it/s]
                       all        141        141      0.938      0.893      0.967      0.913
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        369/500      2.95G     0.2845      0.231     0.8949         10        640: 100% 63/63 [00:18<00:00,  3.46it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.59it/s]
                       all        141        141      0.946      0.888      0.967      0.916
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        370/500      2.95G       0.29     0.2268      0.895         10        640: 100% 63/63 [00:17<00:00,  3.68it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.49it/s]
                       all        141        141      0.947      0.885      0.971      0.917
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        371/500      2.98G     0.2929     0.2312     0.8994         14        640: 100% 63/63 [00:17<00:00,  3.53it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  3.65it/s]
                       all        141        141      0.943      0.876      0.968      0.913
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        372/500      2.99G     0.2982     0.2357     0.9047         11        640: 100% 63/63 [00:17<00:00,  3.68it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.94it/s]
                       all        141        141      0.934      0.874       0.97      0.915
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        373/500         3G     0.2944     0.2322      0.902          5        640: 100% 63/63 [00:17<00:00,  3.67it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  3.07it/s]
                       all        141        141      0.929      0.877      0.972      0.919
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        374/500      3.01G     0.2881     0.2306     0.8922          8        640: 100% 63/63 [00:17<00:00,  3.62it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.89it/s]
                       all        141        141      0.925      0.885      0.976      0.922
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        375/500      3.04G     0.2905     0.2287     0.9007         12        640: 100% 63/63 [00:17<00:00,  3.68it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.93it/s]
                       all        141        141      0.924       0.89      0.975       0.92
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        376/500      3.05G     0.2841     0.2285     0.8948          8        640: 100% 63/63 [00:17<00:00,  3.60it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.89it/s]
                       all        141        141      0.934      0.874      0.975      0.918
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        377/500      3.06G     0.2823     0.2334     0.8925         12        640: 100% 63/63 [00:17<00:00,  3.67it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.81it/s]
                       all        141        141      0.932      0.884      0.972      0.913
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        378/500      3.07G     0.2907     0.2347     0.8929          8        640: 100% 63/63 [00:18<00:00,  3.45it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.37it/s]
                       all        141        141      0.936      0.875       0.97      0.913
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        379/500      3.09G     0.2782      0.223     0.8914         13        640: 100% 63/63 [00:17<00:00,  3.60it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.43it/s]
                       all        141        141      0.925       0.87      0.972      0.914
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        380/500      3.11G     0.2841     0.2299     0.8923          9        640: 100% 63/63 [00:18<00:00,  3.42it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.23it/s]
                       all        141        141      0.903      0.894      0.972      0.915
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        381/500      3.12G     0.2809     0.2265      0.889         10        640: 100% 63/63 [00:17<00:00,  3.67it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.65it/s]
                       all        141        141      0.911      0.894      0.971      0.916
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        382/500      3.13G     0.2881     0.2347     0.8944         11        640: 100% 63/63 [00:17<00:00,  3.51it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  3.16it/s]
                       all        141        141      0.934      0.876      0.972      0.918
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        383/500      3.15G     0.2898     0.2301      0.901          9        640: 100% 63/63 [00:17<00:00,  3.59it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.30it/s]
                       all        141        141      0.941      0.871      0.973      0.922
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        384/500      3.17G     0.2744     0.2267     0.8901         11        640: 100% 63/63 [00:17<00:00,  3.64it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  3.25it/s]
                       all        141        141       0.94      0.872      0.967      0.917
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        385/500      3.18G     0.2891     0.2334      0.893          9        640: 100% 63/63 [00:17<00:00,  3.69it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.21it/s]
                       all        141        141      0.934      0.872      0.968      0.916
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        386/500      3.19G     0.2912     0.2288     0.8977          8        640: 100% 63/63 [00:17<00:00,  3.65it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.40it/s]
                       all        141        141      0.934      0.879      0.963      0.911
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        387/500      3.21G     0.2782     0.2192     0.8899          9        640: 100% 63/63 [00:18<00:00,  3.45it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.54it/s]
                       all        141        141      0.921      0.882      0.961      0.913
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        388/500      3.23G     0.2748     0.2186     0.8945         12        640: 100% 63/63 [00:17<00:00,  3.67it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.96it/s]
                       all        141        141      0.923      0.873      0.961      0.911
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        389/500      3.24G      0.277     0.2232     0.8926          9        640: 100% 63/63 [00:18<00:00,  3.48it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.08it/s]
                       all        141        141      0.926      0.865      0.962      0.915
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        390/500      3.25G     0.2713     0.2176     0.8901          6        640: 100% 63/63 [00:17<00:00,  3.62it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.36it/s]
                       all        141        141      0.893      0.904      0.968      0.918
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        391/500      3.27G     0.2824     0.2217     0.8903          9        640: 100% 63/63 [00:17<00:00,  3.55it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  2.88it/s]
                       all        141        141      0.928      0.867      0.965      0.916
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        392/500      3.29G     0.2877     0.2299      0.896          9        640: 100% 63/63 [00:17<00:00,  3.64it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.31it/s]
                       all        141        141      0.912      0.892      0.965      0.917
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        393/500       3.3G     0.2802     0.2236     0.8914          8        640: 100% 63/63 [00:17<00:00,  3.61it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  3.57it/s]
                       all        141        141      0.888       0.91      0.967      0.917
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        394/500      3.31G     0.2738      0.214     0.8882          8        640: 100% 63/63 [00:17<00:00,  3.58it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.44it/s]
                       all        141        141      0.931      0.875       0.97       0.92
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        395/500      3.33G     0.2836     0.2289     0.9045         11        640: 100% 63/63 [00:17<00:00,  3.62it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.65it/s]
                       all        141        141      0.936      0.876      0.965      0.914
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        396/500      3.35G     0.2776     0.2187     0.9028          8        640: 100% 63/63 [00:17<00:00,  3.54it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.47it/s]
                       all        141        141      0.933       0.88      0.976      0.925
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        397/500      3.36G     0.2848     0.2294     0.8934         10        640: 100% 63/63 [00:17<00:00,  3.68it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.58it/s]
                       all        141        141      0.942      0.881      0.978      0.925
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        398/500      3.37G     0.2861     0.2298     0.8948          8        640: 100% 63/63 [00:18<00:00,  3.48it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.82it/s]
                       all        141        141      0.948      0.872      0.974      0.923
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        399/500      3.39G     0.2866     0.2223     0.8983          9        640: 100% 63/63 [00:17<00:00,  3.67it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.16it/s]
                       all        141        141      0.948      0.865      0.975      0.922
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        400/500      3.41G     0.2705      0.219     0.8901         13        640: 100% 63/63 [00:17<00:00,  3.52it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  2.91it/s]
                       all        141        141      0.947      0.862       0.97      0.916
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        401/500      3.42G     0.2808     0.2222     0.8943         12        640: 100% 63/63 [00:17<00:00,  3.65it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.35it/s]
                       all        141        141      0.949      0.863      0.967      0.917
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        402/500      3.43G     0.2789     0.2248     0.8898          9        640: 100% 63/63 [00:17<00:00,  3.64it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  3.77it/s]
                       all        141        141      0.953      0.867      0.973      0.922
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        403/500      3.45G     0.2682     0.2132     0.8855          9        640: 100% 63/63 [00:19<00:00,  3.30it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.71it/s]
                       all        141        141      0.955      0.867      0.973      0.925
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        404/500      3.46G     0.2717     0.2173     0.8879         11        640: 100% 63/63 [00:17<00:00,  3.63it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.64it/s]
                       all        141        141      0.956      0.866      0.973      0.923
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        405/500      3.48G     0.2817     0.2286     0.8936          6        640: 100% 63/63 [00:19<00:00,  3.18it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.07it/s]
                       all        141        141      0.948      0.864       0.97      0.918
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        406/500      3.49G     0.2799     0.2278     0.8889          6        640: 100% 63/63 [00:18<00:00,  3.42it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.79it/s]
                       all        141        141      0.943      0.863      0.968      0.913
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        407/500      3.51G      0.265     0.2163     0.8877          9        640: 100% 63/63 [00:18<00:00,  3.37it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  3.24it/s]
                       all        141        141      0.939      0.864      0.969      0.914
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        408/500      3.52G     0.2744     0.2263     0.8952         10        640: 100% 63/63 [00:17<00:00,  3.67it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.04it/s]
                       all        141        141       0.91      0.888      0.968      0.917
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        409/500      3.54G     0.2792     0.2322     0.8961          8        640: 100% 63/63 [00:19<00:00,  3.19it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.98it/s]
                       all        141        141      0.934      0.868      0.966      0.913
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        410/500      3.54G     0.2845     0.2314     0.8856         15        640: 100% 63/63 [00:18<00:00,  3.40it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.05it/s]
                       all        141        141      0.933      0.864      0.971      0.919
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        411/500      3.57G     0.2795     0.2214     0.8947          9        640: 100% 63/63 [00:19<00:00,  3.29it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.48it/s]
                       all        141        141      0.941      0.859      0.975      0.927
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        412/500      3.58G     0.2636     0.2038     0.8934          8        640: 100% 63/63 [00:17<00:00,  3.62it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.35it/s]
                       all        141        141       0.93       0.86      0.977      0.927
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        413/500       3.6G     0.2662     0.2127     0.8855          8        640: 100% 63/63 [00:19<00:00,  3.21it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.74it/s]
                       all        141        141      0.936      0.863      0.976      0.924
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        414/500      3.61G     0.2669     0.2139     0.8858          8        640: 100% 63/63 [00:18<00:00,  3.37it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.29it/s]
                       all        141        141      0.937       0.87      0.978      0.927
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        415/500      3.62G     0.2635     0.2043      0.884          4        640: 100% 63/63 [00:19<00:00,  3.24it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.19it/s]
                       all        141        141      0.935      0.874      0.978      0.925
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        416/500      3.64G     0.2749     0.2278     0.8917          9        640: 100% 63/63 [00:16<00:00,  3.71it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  3.72it/s]
                       all        141        141       0.89      0.914      0.978      0.927
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        417/500      3.66G     0.2738     0.2195     0.8867          7        640: 100% 63/63 [00:19<00:00,  3.28it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.19it/s]
                       all        141        141      0.913      0.906      0.978      0.926
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        418/500      3.67G     0.2665     0.2082     0.8926         11        640: 100% 63/63 [00:18<00:00,  3.41it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.23it/s]
                       all        141        141      0.926      0.907      0.976      0.922
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        419/500      3.69G     0.2607     0.2124     0.8849          8        640: 100% 63/63 [00:19<00:00,  3.27it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.44it/s]
                       all        141        141      0.929      0.905      0.974      0.921
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        420/500       3.7G     0.2747     0.2126     0.8953          6        640: 100% 63/63 [00:17<00:00,  3.66it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  3.12it/s]
                       all        141        141      0.927      0.903      0.969      0.916
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        421/500      3.71G      0.273     0.2137     0.8948          7        640: 100% 63/63 [00:18<00:00,  3.47it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.23it/s]
                       all        141        141      0.922        0.9      0.967      0.916
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        422/500      3.72G     0.2638     0.2069     0.8835          8        640: 100% 63/63 [00:18<00:00,  3.46it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.26it/s]
                       all        141        141       0.92      0.901      0.973      0.922
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        423/500      3.75G     0.2648     0.2089     0.8837          8        640: 100% 63/63 [00:19<00:00,  3.20it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.71it/s]
                       all        141        141       0.92      0.902      0.974      0.922
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        424/500      3.76G     0.2667     0.2168     0.8901          8        640: 100% 63/63 [00:17<00:00,  3.63it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  3.20it/s]
                       all        141        141      0.926      0.888      0.973       0.92
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        425/500      3.78G     0.2621     0.2007     0.8862          9        640: 100% 63/63 [00:18<00:00,  3.45it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  3.92it/s]
                       all        141        141      0.915      0.905      0.976      0.923
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        426/500      3.78G     0.2649     0.2176     0.8901          8        640: 100% 63/63 [00:18<00:00,  3.46it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.10it/s]
                       all        141        141      0.913      0.907      0.972      0.921
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        427/500      3.81G     0.2676     0.2167     0.8968         10        640: 100% 63/63 [00:19<00:00,  3.19it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.21it/s]
                       all        141        141       0.92      0.906      0.979      0.925
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        428/500      3.82G     0.2572      0.208     0.8859         12        640: 100% 63/63 [00:17<00:00,  3.51it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  3.26it/s]
                       all        141        141      0.922      0.909      0.976      0.923
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        429/500      3.83G     0.2727     0.2235      0.889          8        640: 100% 63/63 [00:18<00:00,  3.43it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.01it/s]
                       all        141        141       0.92      0.913      0.977      0.922
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        430/500      3.84G     0.2671     0.2082     0.8924          7        640: 100% 63/63 [00:17<00:00,  3.63it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.96it/s]
                       all        141        141      0.921      0.912      0.977      0.921
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        431/500      3.87G     0.2581     0.2053     0.8811         11        640: 100% 63/63 [00:19<00:00,  3.19it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.89it/s]
                       all        141        141      0.924      0.913      0.976      0.923
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        432/500      3.88G     0.2611     0.2078     0.8863          9        640: 100% 63/63 [00:18<00:00,  3.44it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.59it/s]
                       all        141        141      0.922      0.912      0.976      0.921
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        433/500      3.89G     0.2659     0.2093     0.8884          6        640: 100% 63/63 [00:18<00:00,  3.45it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.28it/s]
                       all        141        141      0.918      0.911      0.974       0.92
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        434/500       3.9G     0.2633     0.2141     0.8899          6        640: 100% 63/63 [00:18<00:00,  3.40it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.67it/s]
                       all        141        141      0.923      0.907      0.968      0.914
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        435/500      3.92G     0.2625     0.2073     0.8857         12        640: 100% 63/63 [00:19<00:00,  3.22it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.75it/s]
                       all        141        141      0.939      0.893      0.967      0.913
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        436/500      3.94G     0.2716     0.2133     0.8916          7        640: 100% 63/63 [00:18<00:00,  3.42it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.68it/s]
                       all        141        141       0.94      0.886      0.968      0.914
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        437/500      3.95G     0.2637     0.2067     0.8844         15        640: 100% 63/63 [00:18<00:00,  3.42it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.79it/s]
                       all        141        141      0.945      0.878      0.974      0.925
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        438/500      3.96G     0.2627     0.2104     0.8907         15        640: 100% 63/63 [00:18<00:00,  3.41it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.45it/s]
                       all        141        141      0.947      0.875      0.974      0.924
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        439/500      3.98G     0.2679     0.2156     0.8868          9        640: 100% 63/63 [00:19<00:00,  3.21it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.08it/s]
                       all        141        141      0.949      0.876      0.974      0.923
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        440/500         4G     0.2581     0.1997     0.8888         11        640: 100% 63/63 [00:18<00:00,  3.36it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.93it/s]
                       all        141        141      0.919      0.912      0.977      0.926
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        441/500      4.01G     0.2561     0.1988     0.8903         11        640: 100% 63/63 [00:19<00:00,  3.21it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.84it/s]
                       all        141        141      0.947      0.884      0.976      0.927
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        442/500      4.02G     0.2672     0.2044     0.8942          5        640: 100% 63/63 [00:17<00:00,  3.63it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.39it/s]
                       all        141        141      0.915      0.919      0.974      0.924
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        443/500      4.04G     0.2571     0.1945      0.883         14        640: 100% 63/63 [00:19<00:00,  3.18it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.97it/s]
                       all        141        141      0.947      0.882      0.971      0.921
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        444/500      4.06G     0.2559     0.2059     0.8756         11        640: 100% 63/63 [00:18<00:00,  3.40it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.20it/s]
                       all        141        141      0.946      0.878      0.978      0.928
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        445/500      4.07G     0.2607     0.2091     0.8884         11        640: 100% 63/63 [00:19<00:00,  3.19it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.24it/s]
                       all        141        141      0.948      0.876      0.975      0.925
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        446/500      4.08G      0.265     0.2058     0.8908         11        640: 100% 63/63 [00:17<00:00,  3.58it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.60it/s]
                       all        141        141      0.946      0.877      0.974      0.927
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        447/500       4.1G     0.2514     0.2011     0.8806         10        640: 100% 63/63 [00:19<00:00,  3.17it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.81it/s]
                       all        141        141      0.944      0.885      0.978      0.929
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        448/500      4.12G     0.2499     0.2009     0.8845         12        640: 100% 63/63 [00:18<00:00,  3.40it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.90it/s]
                       all        141        141      0.947      0.878      0.978      0.927
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        449/500      4.13G     0.2598     0.2027      0.886         12        640: 100% 63/63 [00:19<00:00,  3.19it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.47it/s]
                       all        141        141      0.947      0.878      0.978      0.926
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        450/500      4.14G     0.2571     0.2063     0.8894         12        640: 100% 63/63 [00:17<00:00,  3.66it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.08it/s]
                       all        141        141      0.925      0.903      0.976      0.925
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        451/500      4.16G     0.2622     0.2052     0.8882          3        640: 100% 63/63 [00:18<00:00,  3.36it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  4.48it/s]
                       all        141        141      0.926      0.906      0.975      0.923
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        452/500      4.18G     0.2503     0.1963     0.8819          9        640: 100% 63/63 [00:18<00:00,  3.35it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.12it/s]
                       all        141        141      0.925      0.901      0.978      0.925
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        453/500      4.19G     0.2577     0.2038     0.8861         11        640: 100% 63/63 [00:20<00:00,  3.14it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.52it/s]
                       all        141        141      0.892      0.922      0.977      0.925
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        454/500       4.2G       0.25     0.2014      0.888         12        640: 100% 63/63 [00:19<00:00,  3.31it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:00<00:00,  5.41it/s]
                       all        141        141       0.94       0.87      0.975      0.924
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        455/500      4.22G     0.2567      0.204     0.8863         11        640: 100% 63/63 [00:18<00:00,  3.34it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  3.24it/s]
                       all        141        141       0.88      0.925      0.975      0.924
    [34m[1mEarlyStopping: [0mTraining stopped early as no improvement observed in last 100 epochs. Best results observed at epoch 355, best model saved as best.pt.
    To update EarlyStopping(patience=100) pass a new patience value, i.e. `patience=300` or use `patience=0` to disable EarlyStopping.
    
    455 epochs completed in 2.401 hours.
    Optimizer stripped from runs/detect/train/weights/last.pt, 5.5MB
    Optimizer stripped from runs/detect/train/weights/best.pt, 5.5MB
    
    Validating runs/detect/train/weights/best.pt...
    Ultralytics 8.3.179 🚀 Python-3.11.13 torch-2.6.0+cu124 CUDA:0 (Tesla T4, 15095MiB)
    YOLO11n summary (fused): 100 layers, 2,587,222 parameters, 0 gradients, 6.3 GFLOPs
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:02<00:00,  2.17it/s]
                       all        141        141      0.948      0.897      0.987      0.939
                         A          4          4          1      0.777      0.995      0.939
                         B          9          9          1      0.672      0.973      0.903
                         C          3          3      0.952      0.667      0.913      0.834
                         D          6          6      0.995          1      0.995      0.995
                         E          4          4       0.98          1      0.995      0.995
                         F          8          8       0.86      0.875      0.971       0.95
                         G          5          5      0.845          1      0.995      0.995
                         H          8          8          1      0.893      0.995      0.947
                         I          2          2          1      0.546      0.995      0.995
                         J          8          8      0.994          1      0.995      0.704
                         K          6          6      0.984      0.833      0.972      0.919
                         L          4          4      0.787          1      0.995      0.971
                         M          8          8          1      0.889      0.995      0.922
                         N          3          3      0.744          1      0.995      0.909
                         O          7          7          1      0.764      0.995      0.972
                         P          7          7      0.999          1      0.995      0.917
                         Q          4          4      0.982          1      0.995      0.923
                         R          7          7          1       0.89      0.995      0.965
                         S          4          4      0.984          1      0.995      0.995
                         T          6          6      0.984      0.833      0.972      0.961
                         U          7          7      0.871          1      0.995      0.995
                         V          5          5          1      0.813      0.995      0.995
                         W          3          3      0.961          1      0.995      0.995
                         X          1          1      0.733          1      0.995      0.895
                         Y          8          8      0.982      0.875      0.949      0.876
                         Z          4          4          1          1      0.995      0.958
    Speed: 0.2ms preprocess, 3.5ms inference, 0.0ms loss, 3.9ms postprocess per image
    Results saved to [1mruns/detect/train[0m
    💡 Learn more at https://docs.ultralytics.com/modes/train



```python
from IPython.display import Image as IPyImage

# Display the confusion matrix image from the specified directory in Kaggle
IPyImage(filename='/content/runs/detect/train/confusion_matrix.png', width=1000)
```


```python
IPyImage(filename=f'/content/runs/detect/train/results.png', width=1000)
```


```python
IPyImage(filename=f'/content/runs/detect/train/val_batch0_pred.jpg', width=1000)
```

# **Validation Of The Model**





```python
# Run the validation task using YOLO
!yolo task=detect mode=val model=/content/runs/detect/train/weights/best.pt data=/content/dataset/data.yaml
```

    Ultralytics 8.3.179 🚀 Python-3.11.13 torch-2.6.0+cu124 CUDA:0 (Tesla T4, 15095MiB)
    YOLO11n summary (fused): 100 layers, 2,587,222 parameters, 0 gradients, 6.3 GFLOPs
    [34m[1mval: [0mFast image access ✅ (ping: 0.0±0.0 ms, read: 1360.7±255.3 MB/s, size: 30.3 KB)
    [34m[1mval: [0mScanning /content/dataset/valid/labels.cache... 141 images, 0 backgrounds, 0 corrupt: 100% 141/141 [00:00<?, ?it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 9/9 [00:02<00:00,  4.19it/s]
                       all        141        141      0.929      0.909      0.986      0.939
                         A          4          4      0.949       0.75      0.945      0.926
                         B          9          9          1       0.89      0.995      0.891
                         C          3          3      0.979          1      0.995      0.913
                         D          6          6       0.88      0.833      0.972      0.906
                         E          4          4      0.789          1      0.995      0.995
                         F          8          8      0.955          1      0.995      0.974
                         G          5          5       0.92          1      0.995      0.972
                         H          8          8          1      0.988      0.995      0.954
                         I          2          2          1      0.535      0.995      0.995
                         J          8          8      0.993          1      0.995      0.685
                         K          6          6          1      0.696      0.995      0.946
                         L          4          4      0.768      0.838      0.945      0.925
                         M          8          8          1      0.893      0.995      0.922
                         N          3          3      0.735          1      0.995      0.995
                         O          7          7      0.989          1      0.995      0.995
                         P          7          7      0.986      0.857      0.978      0.892
                         Q          4          4      0.971          1      0.995      0.921
                         R          7          7      0.759          1      0.978      0.952
                         S          4          4      0.975          1      0.995      0.995
                         T          6          6      0.959      0.667      0.948      0.932
                         U          7          7      0.938          1      0.995      0.975
                         V          5          5          1      0.808      0.995      0.964
                         W          3          3      0.933          1      0.995      0.995
                         X          1          1      0.725          1      0.995      0.895
                         Y          8          8      0.982      0.875      0.962      0.913
                         Z          4          4      0.976          1      0.995      0.972
    Speed: 1.0ms preprocess, 4.3ms inference, 0.0ms loss, 3.7ms postprocess per image
    Results saved to [1mruns/detect/val[0m
    💡 Learn more at https://docs.ultralytics.com/modes/val


# **Prediction with Random Images**


```python
import glob
import os
from IPython.display import Image as IPyImage, display

# Get the latest prediction folder for detection in Kaggle
latest_folder = max(glob.glob('/content/runs/detect/predict*/'), key=os.path.getmtime)

# Display images from the prediction folder
for img in glob.glob(f'{latest_folder}/*.jpg')[15:18]:
    display(IPyImage(filename=img, width=300))
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    /tmp/ipython-input-1005507424.py in <cell line: 0>()
          4 
          5 # Get the latest prediction folder for detection in Kaggle
    ----> 6 latest_folder = max(glob.glob('/content/runs/detect/predict*/'), key=os.path.getmtime)
          7 
          8 # Display images from the prediction folder


    ValueError: max() arg is an empty sequence


# **Predictions on Videos**



```python
# Input video path for the first video in Kaggle
input_video_path = "/kaggle/input/asl-videos/asl_video1_40sn.mp4"  # First video path
# Output paths for saving the prediction result
output_video_path = "/kaggle/working/runs/detect/predict/asl_video1_40sn_output.avi"  # YOLO default output in .avi format

# Run YOLO on the first video for object detection
!yolo task=detect mode=predict model="/kaggle/working/runs/detect/train/weights/best.pt" conf=0.25 source="{input_video_path}" save=True

# Results saved to runs/detect/predict2
#💡 Learn more at https://docs.ultralytics.com/modes/predict
```


```python
# Check if the .mp4 file is successfully created
!ls /kaggle/working/runs/detect/predict2/
```


```python
# Input video path for the second video in Kaggle
input_video_path = "/kaggle/input/asl-videos/asl_video2_30sn.mp4"  # Second video path
# Output paths for saving the prediction result
output_video_path = "/kaggle/working/runs/detect/predict/asl_video2_30sn_output.avi"  # YOLO default output in .avi format

# Run YOLO on the second video for object detection
!yolo task=detect mode=predict model="/kaggle/working/runs/detect/train/weights/best.pt" conf=0.25 source="{input_video_path}" save=True
```


```python
# Check if the .mp4 file is successfully created
!ls /kaggle/working/runs/detect/predict3/
```
