# Object Detection을 위한 신경망 모델의 Feature Map 분석
* 본 프로젝트에선 Object Detection 모델의 다양한 layer의 feature map을 열화시켰을 때 object detection에 어떤 영향을 끼쳤는지에 대해 연구하였다.
* bilinearInterplate 기법을 사용하여 feature map을 열화시켰다.
   * 이때 압축 scale이 4라는 것은, feature map을 1/4배로 축소시킨 후, 기존 feature map 크기로 복구시켰다는 것을 의미한다.
## About Codes
```
train_net.py --cr int int int int int int (왼쪽부터 p2~p6의 압축 scale)
ex) train_net.py --cr 1 1 1 1 1 4 (p6만 압축 scale 4, p2 ~ p5는 scale 1 (열화x))
```

다양한 조합에 대해 실험을 진행해야 하기 때문에 각 feature map에 대한 압축 scale을 작성해주면, 입력 scale 대로 열화하여 성능 확인 및 json파일로 결과를 저장하도록 코드를 변경해주었다.
* [feature map 압축](https://github.com/dbwp031/detectron2/blob/main/detectron2/modeling/proposal_generator/rpn.py#L455)
```
        for f, ratio in zip(self.in_features,self.compress_ratio):
            h,w = features[f].shape[-2:]
            if ratio == 1:
                up_sample = features[f]
            else:
                down_sample = torch.nn.functional.interpolate(features[f], (int(h//ratio), int(w//ratio)),mode='bilinear')
                up_sample = torch.nn.functional.interpolate(down_sample, (int(h),int(w)),mode='bilinear')
            new_features.append(up_sample)
        features = new_features
```
* [json 저장](https://github.com/dbwp031/detectron2/blob/main/tools/train_net.py#L149)
```
        if comm.is_main_process():
            verify_results(cfg, res)
            # expn = f"2-{args.cr[0]}_3-{args.cr[1]}_4-{args.cr[2]}-5_{args.cr[3]}-6_{args.cr[4]}"
            result = [vars(args),res]
            expn = f"final_p4-3_p3_{args.cr[0]}-p5_{args.cr[1]}-mix"
            with open(f'/root/workspace/test_d/output/cr_output/{expn}.json', 'w') as outfile:
                # json.dump(vars(args),outfile,indent=4)
                # json.dump(res,outfile, indent=4)
                json.dump(result,outfile, indent=4)
        return res
```
## Experiment Descriptions

### Proposed Method
  * Feature Pyramid에서 물체 판별에 큰 영향이 있는 key feature maps만 전달함.
  * 그 후, key feature maps의 정보를 활용하여 feature pyramid를 복구함.
![image](https://user-images.githubusercontent.com/65337423/206092448-87e6da2b-96b9-4441-bc49-7106b9fc1569.png)

### 압축률에 따른 object scale 별 Average Precision
![image](https://user-images.githubusercontent.com/65337423/206092635-191e42c9-53d9-4f71-ac69-ceaab92f8337.png)
  * 각 Object 별 성능 영향도가 높은 Feature Map
    * Small Object: P2, P3
    * Medium Object: P4, P5
    * Large Object: P5, P6
### 물체 크기별 영향도가 높은 Feature Map들을 최대 열화시켰을 때의 성능
![image](https://user-images.githubusercontent.com/65337423/206092702-ccf2660a-5eb5-4ac9-a1cb-f6cf6e61a744.png)

* 각 Object Scale에 크게 관여하는 두 Feature Map을 모두 최대 열화시켰을 때, 상당한 성능 drop.
  * Small Object, Medium Object: -10AP
  * Large Object: -1.8AP 
  * (P2,P3), (P4,P5), (P5,P6) 조합으로 제거하면 상당한 성능 drop을 가지게 됨.


### Key Feature Map 분석
![image](https://user-images.githubusercontent.com/65337423/206092744-15de1215-1bb6-416a-9d93-cc158139d449.png)

## Conclusion
  * 본 연구에서는 Feature Pyramid의 각 Feature들을 열화시켜 해당 Feature의 실제 모델 성능의 영향도를 분석함.
  * 분석을 통해, 물체 사이즈에 따라 영향을 받는 Feature Map이 다르다는 것을 파악하였고, 모든 물체 사이즈에 대한 판별을 고려하여 Key Feature Map을 추출하였음.
  * 이후, Key Feature Map만을 활용하여 전달되지 않은 feature map을 대체해 Feature Pyramid를 복구시켜 1.4% AP drop으로 7.96배의 데이터량을 감소시킴.

