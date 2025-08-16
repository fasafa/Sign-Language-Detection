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
    
         ......
    
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
# Run YOLO on the first video for object detection
!yolo task=detect mode=predict model=best.pt" conf=0.25 source="image.jpg" save=True

# Results saved to runs/detect/predict

```


# **Predictions on Videos**
```python
# Run YOLO on the first video for object detection
!yolo task=detect mode=predict model=best.pt" conf=0.25 source="videos.mp4" save=True

# Results saved to runs/detect/predict

```






