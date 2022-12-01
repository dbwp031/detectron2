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
