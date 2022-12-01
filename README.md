# Object Detection을 위한 신경망 모델의 Feature Map 분석
* 본 프로젝트에선 Object Detection 모델의 다양한 layer의 feature map을 열화시켰을 때 object detection에 어떤 영향을 끼쳤는지에 대해 연구하였다.
* bilinearInterplate 기법을 사용하여 feature map을 열화시켰다.
   * 이때 압축 scale이 4라는 것은, feature map을 1/4배로 축소시킨 후, 기존 feature map 크기로 복구시켰다는 것을 의미한다.
## about codes
```
train_net.py --cr int int int int int int (왼쪽부터 p2~p6의 압축 scale)
ex) train_net.py --cr 1 1 1 1 1 4 (p6만 압축 scale 4, p2 ~ p5는 scale 1 (열화x))
```

다양한 조합에 대해 실험을 진행해야 하기 때문에 각 feature map에 대한 압축 scale을 작성해주면, 입력 scale 대로 열화하여 성능 확인 및 json파일로 결과를 저장하도록 코드를 변경해주었다.
