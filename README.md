### IJCAI-PRCAI2020 3D AI Challenge: Instance Segmentation
A榜：8/559 B榜：6/559
![Results](https://github.com/rechardgu0816/Instance-segmentation-tianchi/blob/master/images/1.jpg)
# Improvements
1、Data Augmentation

2、Baseline(Hybrid Task Cascade)

3、tricks：GA-RPN、DCN、SyncBn etc
# Experiment
Install [mmdetectionv2](https://github.com/open-mmlab/mmdetection)

Train
```python
./tools/dist_train.sh configs/ga_htc_iou.py 2
```
Pushlish model

```python
python tools/publish_model.py work_dirs//ga_htc_iou/lasted.pth checkpoint/ga_htc_iou.pth
```
Test
```python
./tools/dist_test.sh configs/ga_htc_iou.py checkpoint/ga_htc_iou-52d8cd38.pth --format-only --options "jsonfile_prefix=./seg_test_results" 2
```
