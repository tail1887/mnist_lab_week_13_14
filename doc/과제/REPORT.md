# MNIST 손글씨 숫자 인식 리포트

## 0. 반·팀원

- 반: 302반
- 팀: 3팀
- 팀원: 박태정, 유중일, 여서진

## 1. 실험 목적

이번 과제는 PyTorch, TensorFlow 같은 딥러닝 프레임워크 없이 **NumPy만으로 MNIST 손글씨 숫자 분류 신경망을 직접 구현**하는 것을 목표로 한다.

28x28 크기의 손글씨 숫자 이미지를 784차원 벡터로 입력받아 0~9 중 하나의 클래스로 분류한다. 단순히 정확도를 높이는 것뿐 아니라, 다음 흐름을 직접 구현하고 이해하는 데 목적이 있다.

- Forward propagation
- Loss 계산
- Backward propagation
- Optimizer를 이용한 parameter update

최종 목표는 테스트 정확도 95% 이상이며, 권장 목표는 97% 이상이다.

## 2. 모델 구조

### 2.1 전체 구조


| 구분    | 구성                                         |
| ----- | ------------------------------------------ |
| 입력층   | 784차원 입력                                   |
| 은닉층 1 | Affine(`512`) → BatchNorm → ReLU → Dropout |
| 은닉층 2 | Affine(`256`) → BatchNorm → ReLU → Dropout |
| 출력층   | Affine(10) → Softmax                       |


실제 실험에 사용한 구조:

```text
입력 784
-> Affine(512)
-> BatchNorm
-> ReLU
-> Dropout
-> Affine(256)
-> BatchNorm
-> ReLU
-> Dropout
-> Affine(10)
-> Softmax
```

### 2.2 구현한 주요 구성 요소


| 구성 요소                | 파일                   | 역할                |
| -------------------- | -------------------- | ----------------- |
| `ReLU`               | `src/activations.py` | 은닉층의 비선형 활성화 함수   |
| `Softmax`            | `src/activations.py` | 출력값을 클래스별 확률로 변환  |
| `Affine`             | `src/layers.py`      | 선형 변환             |
| `BatchNorm`          | `src/layers.py`      | 배치 정규화를 통한 학습 안정화 |
| `Dropout`            | `src/layers.py`      | 과적합 완화            |
| `cross_entropy_loss` | `src/losses.py`      | 다중 클래스 분류 손실 계산   |
| `SGD`, `Adam`        | `src/optimizers.py`  | 파라미터 업데이트         |
| `NeuralNetwork`      | `src/network.py`     | 전체 신경망 구성         |


## 3. 개발 순서 및 테스트 기록

과제 요구사항에 따라 각 기능을 단계별로 구현하고, 해당 단계의 테스트를 통과하는지 확인한다. 테스트가 실패한 경우에는 원인을 분석한 뒤 수정하고 다시 테스트한다.

### 3.1 Step 1: ReLU

- 구현 대상: `ReLU.forward`, `ReLU.backward`
- 수정 파일: `src/activations.py`
- 테스트 명령: `pytest tests/test_relu.py -v`
- 코드 스크린샷: 
<img src="image/스크린샷 2026-05-25 오후 1.54.11.png" alt="스크린샷" width="700">

- 테스트 결과 스크린샷: 
<img src="image/스크린샷 2026-05-25 오후 1.55.55.png" alt="스크린샷" width="700">

- 수정 내용 및 확인한 점: 
레루 함수에서 들어오는 x는 affine forward함수의 리턴값이다.
역전파에서 마스크값이 0인 a와 연결된 모든 노드는 막힌다
레루함수를 적용하는김에 마스크를 만들고 이 마스크는 역전파에서 신호가 없는 노드를 표현하기 위해서 활용한다.

### 3.2 Step 2: Softmax

- 구현 대상: `Softmax.forward`, `Softmax.backward`
- 수정 파일: `src/activations.py`
- 테스트 명령: `pytest tests/test_softmax.py -v`
- 코드 스크린샷: 
<img src="image/스크린샷 2026-05-25 오후 2.01.29.png" alt="스크린샷" width="700">

- 테스트 결과 스크린샷: 
<img src="image/스크린샷 2026-05-25 오후 2.03.04.png" alt="스크린샷" width="700">

- 수정 내용 및 확인한 점: 
np.max(x)를 그냥 사용해도 테스트가 통과하는데 이부분은 입력이 1차원 벡터면 상관없지만 그 외에는 문제가 발생하게 된다. 
예: 사진이 여러장일 때, 각각의 최댓값이 아닌 하나의 최댓값을 공유하게됨

### 3.3 Step 3: Affine

- 구현 대상: `Affine.forward`, `Affine.backward`
- 수정 파일: `src/layers.py`
- 테스트 명령: `pytest tests/test_affine.py -v`
- 코드 스크린샷:
<img src="image/스크린샷 2026-05-25 오후 2.47.26.png" alt="스크린샷" width="700">

- 테스트 결과 스크린샷: 
<img src="image/스크린샷 2026-05-25 오후 2.49.33.png" alt="스크린샷" width="700">

- 수정 내용 및 확인한 점: 
식, x @ W + b에 대해
x를 기준으로 편미분하면 -> W
W를 기준으로 편미분하면 -> x
b를 기준으로 편미분하면 -> 1
각 변수,
dx는 x값이 loss에 미치는 영향
dW는 가중치가 loss에 미치는 영향
db는 b(편향)이 loss에 미치는 영향이다.

### 3.4 Step 4: Cross Entropy Loss

- 구현 대상: `cross_entropy_loss`
- 수정 파일: `src/losses.py`
- 테스트 명령: `pytest tests/test_cross_entropy_loss.py -v`
- 코드 스크린샷:
<img src="image/스크린샷 2026-05-25 오후 3.46.35.png" alt="스크린샷" width="700">

- 테스트 결과 스크린샷:
<img src="image/스크린샷 2026-05-25 오후 3.47.39.png" alt="스크린샷" width="700">

- 수정 내용 및 확인한 점:
교차 엔트로피 오차는 소프트 맥스 함수와 환상의 짝궁이다. 역전파 과정에서 교차 엔트로피의 미분값이랑 소프트맥스의 미분값이랑 곱하면 항들이 정리되어 식이 깔끔해진다. 

### 3.5 Step 5: SGD

- 구현 대상: `SGD.update`
- 수정 파일: `src/optimizers.py`
- 테스트 명령: `pytest tests/test_sgd.py -v`
- 코드 스크린샷: 
<img src="image/스크린샷 2026-05-25 오후 4.29.14.png" alt="스크린샷" width="700">

- 테스트 결과 스크린샷: 
<img src="image/스크린샷 2026-05-25 오후 4.30.29.png" alt="스크린샷" width="700">

- 수정 내용 및 확인한 점: 
원래 가중치에 그 가중치에 대한 미분값에 러닝레이트를 곱한 값을 빼면 된다.. 러닝메이트가 너무 크면 최소값을 지나치고 너무 작으면 학습이 오래걸리고 갱신이 안된다.

### 3.6 Step 6: Adam

- 구현 대상: `Adam.update`
- 수정 파일: `src/optimizers.py`
- 테스트 명령: `pytest tests/test_adam.py -v`
- 코드 스크린샷: 
<img src="image/스크린샷 2026-05-25 오후 4.49.27.png" alt="스크린샷" width="700">

- 테스트 결과 스크린샷: 
<img src="image/스크린샷 2026-05-25 오후 4.48.54.png" alt="스크린샷" width="700">

- 수정 내용 및 확인한 점: 
기존의 SGD는 모든 파라미터에 동일한 고정 learning_rate를 적용하므로, 기울기 크기나 방향 변화에 따라 갱신 폭을 조절하지 못해 탐색 경로가 비효율적일 수 있다.
Adam은 기울기의 1차 모멘트(m)와 2차 모멘트(v)를 누적하고, 초기값이 0이라 작게 추정되는 문제를 편향 보정으로 보완한 뒤 파라미터별 업데이트 크기를 조정한다. 이를 통해 SGD보다 손실이 더 안정적으로 감소하는 것을 확인했다.
t값이 0일 때 강하게 보정하고 학습이 충분히 진행되면 t값이 커져 보정효과가 점점 줄어든다.

### 3.7 Step 7: NeuralNetwork

- 구현 대상: `NeuralNetwork`
- 수정 파일: `src/network.py`
- 테스트 명령: `pytest tests/test_neural_network.py -v`
- 코드 스크린샷:
<img src="image/스크린샷 2026-05-25 오후 5.37.49.png" alt="스크린샷" width="700">
<img src="image/스크린샷 2026-05-25 오후 5.38.20.png" alt="스크린샷" width="700">

- 테스트 결과 스크린샷:
<img src="image/스크린샷 2026-05-25 오후 5.42.11.png" alt="스크린샷" width="700">

- 수정 내용 및 확인한 점:
후반 부분을 먼저 완성해야 테스트가 실행가능하다.

### 3.8 Step 8: BatchNorm

- 구현 대상: `BatchNorm.forward`, `BatchNorm.backward`
- 수정 파일: `src/layers.py`
- 테스트 명령: `pytest tests/test_batchnorm.py -v`
- 코드 스크린샷: 
<img src="image/스크린샷 2026-05-25 오후 8.07.39.png" alt="스크린샷" width="700">
<img src="image/스크린샷 2026-05-25 오후 8.07.52.png" alt="스크린샷" width="700">

- 테스트 결과 스크린샷: 
<img src="image/스크린샷 2026-05-25 오후 8.38.51.png" alt="스크린샷" width="700">

- 수정 내용 및 확인한 점: 
배치놈은 affine레이어 이후에 relu에 값을 넣기전에 데이터를 정규화해주는 레이어다. 평균 0, 분산 1로 값을 바꿈으로써 데이터 분포를 깔끔하게 해준다. 여기서 중요한점은 모델이 학습가능한 가중치를 부여함으로서 최적의 위치로 이동할수 있게 자율권을 준다는점이다. 그래서 미분을 할때 도 두개의 레이어로 분리해서 생각하면 깔끔하게 계산할수 있따.

### 3.9 Step 9: Dropout

- 구현 대상: `Dropout.forward`, `Dropout.backward`
- 수정 파일: `src/layers.py`
- 테스트 명령: `pytest tests/test_dropout.py -v`
- 코드 스크린샷: 
<img src="image/스크린샷 2026-05-25 오후 9.01.25.png" alt="스크린샷" width="700">

- 테스트 결과 스크린샷: 
<img src="image/스크린샷 2026-05-25 오후 9.02.58.png" alt="스크린샷" width="700">

- 수정 내용 및 확인한 점: 
드롭아웃은 과적합 방지를 위한 정규화 층이다. 학습 중 활성값 일부를 랜덤하게 0으로 만들어, 모델이 특정 뉴런이나 특정 특징 조합에 과하게 의존하지 않도록 만든다. 순전파에서 0으로 꺼진 위치는 역전파에서도 같은 마스크에 의해 gradient가 전달되지 않는다.
BatchNorm과 Dropout은 목적이 다르기 때문에 함께 사용할 수 있지만, 항상 같이 쓰는 것은 아니다. Dropout이 활성값을 랜덤하게 0으로 만들면 BatchNorm이 관찰하는 분포가 흔들릴 수 있으므로, 일반적으로는 `Affine -> BatchNorm -> ReLU` 순서로 학습을 안정화하고, 과적합이 여전히 심할 때 뒤쪽 fully-connected layer 근처에 Dropout을 추가한다.

### 3.10 Step 10: Train

- 구현 대상: `train`
- 수정 파일: `src/training.py`
- 테스트 명령: `pytest tests/test_training.py -v`
- 코드 스크린샷:
<img src="image/스크린샷 2026-05-25 오후 9.29.16.png" alt="스크린샷" width="700">

- 테스트 결과 스크린샷:
<img src="image/스크린샷 2026-05-25 오후 9.28.04.png" alt="스크린샷" width="700">

- 수정 내용 및 확인한 점:
미니배치 학습 루프에서 매 epoch마다 데이터를 섞고, 배치 단위로 forward → loss 계산 → backward → optimizer update가 수행되도록 구현했다.
역전파 부분에서 소프트맥스와 크로스 엔트로피의 미분값을 곱한 공식을 사용해서 깔끔한 공식으로 바로 역전파를 시작할수 있었다.

전체 테스트:

```bash
pytest tests/ -v
```

**테스트 과정에서 발견한 문제와 수정 내용**

- `train` 함수에서 `loss_history`가 초기화되지 않아 학습 손실을 저장할 수 없던 문제를 수정했다.
- 마지막 mini-batch의 크기가 `batch_size`보다 작을 수 있으므로, 역전파 시작 gradient 계산에 실제 배치 크기인 `current_batch_size`를 사용하도록 수정했다.
- epoch마다 마지막 batch loss만 저장하지 않고, 전체 batch loss를 누적해 epoch 평균 loss를 저장하도록 수정했다.


## 4. 실험 환경


| 항목             | 내용                                    |
| -------------- | ------------------------------------- |
| Python version | 3.11                                  |
| 사용 라이브러리       | NumPy, math, random, time, matplotlib |
| 실행 환경          | 로컬 Python 가상환경                       |
| 하드웨어           | CPU                                   |
| 데이터셋           | MNIST                                 |


## 5. 실험 진행 방식

이번 실험은 한 번에 최종 모델을 만든 것이 아니라, 기본 구현을 검증한 뒤 학습 결과를 분석하고 설정값을 바꾸며 성능을 개선하는 방식으로 진행했다.

실험 흐름:

1. 기본 모델로 학습이 정상적으로 진행되는지 확인한다.
2. `SGD`로 baseline 성능을 측정한다.
3. loss와 accuracy를 보고 문제점을 분석한다.
4. learning rate, epoch, dropout 등을 조정한다.
5. `Adam`으로 optimizer를 변경해 수렴 속도와 정확도를 비교한다.
6. 가장 좋은 설정을 최종 모델로 선택한다.

## 6. 실험 기록

### 6.1 SGD 실험

#### 6.1.1 SGD-1: 기본 설정 실험


| 항목                    | 값                           |
| --------------------- | --------------------------- |
| Optimizer             | SGD                         |
| Learning rate         | `0.001`                     |
| Epochs                | `20`                        |
| Batch size            | `128`                       |
| Hidden layer          | `784 -> 512 -> 256 -> 10`   |
| BatchNorm             | 사용                         |
| Dropout               | 사용 (`drop_ratio=0.5`)      |


**실험 의도**

먼저 SGD로 기본 학습이 정상적으로 진행되는지 확인한다. BatchNorm과 Dropout을 모두 사용한 상태에서 loss가 epoch에 따라 감소하는지, test accuracy가 어느 정도까지 올라가는지 확인한다.

**결과**


| 항목            | 결과     |
| ------------- | ------ |
| Train loss    | `0.7770` |
| Test accuracy | `86.87%` |
| 총 파라미터 수      | `537,354` |


**스크린샷**

- 학습 로그: 
<img src="image/스크린샷 2026-05-25 오후 10.33.05.png" alt="스크린샷" width="700">

- Loss curve: 
<img src="image/스크린샷 2026-05-25 오후 10.34.55.png" alt="스크린샷" width="700">

**분석**

SGD 기본 설정에서도 loss가 `2.3965`에서 `0.7770`까지 꾸준히 감소하여 학습은 정상적으로 진행되었다. 최종 train accuracy는 `86.46%`, test accuracy는 `86.87%`로 두 값이 비슷하므로 과적합은 크게 보이지 않았지만, 정확도 자체는 아직 개선 여지가 있다.

**다음 실험에서 바꿀 점**

다음 실험에서는 epoch를 조정하여 SGD의 수렴 속도와 최종 정확도가 개선되는지 확인한다.

---

#### 6.1.2 SGD-2: epoch 조정


| 항목                    | 값                           |
| --------------------- | --------------------------- |
| Optimizer             | SGD                         |
| Learning rate         | `0.001`                     |
| Epochs                | `40`                        |
| Batch size            | `128`                       |
| Hidden layer          | `784 -> 512 -> 256 -> 10`   |
| BatchNorm             | 사용                         |
| Dropout               | 사용 (`drop_ratio=0.5`)      |


**실험 의도**

이전 실험의 결과를 바탕으로 epoch을 조정해 loss 감소와 정확도 변화를 확인한다.
가설: 이전 실험에서는 epoch 수가 부족해 모델이 충분히 학습하지 못했을 가능성이 있다. 따라서 이번 실험에서는 epoch 수를 늘려 학습을 더 진행했을 때 loss가 추가로 감소하고 정확도가 개선되는지 확인한다.
**결과**


| 항목            | 결과     |
| ------------- | ------ |
| Train loss    | `0.5479` |
| Test accuracy | `90.33%` |
| 총 파라미터 수      | `537,354` |


**스크린샷**

- 학습 로그: 
<img src="image/스크린샷 2026-05-25 오후 10.41.25.png" alt="스크린샷" width="700">

- Loss curve: 
<img src="image/스크린샷 2026-05-25 오후 10.41.45.png" alt="스크린샷" width="700">

**분석**

epoch을 `20`에서 `40`으로 늘리자 loss가 `0.7770`에서 `0.5479`까지 더 감소했고, test accuracy도 `86.87%`에서 `90.33%`로 향상되었다. train accuracy와 test accuracy가 각각 `89.81%`, `90.33%`로 비슷하게 유지되어 과적합보다는 학습이 더 진행되면서 성능이 개선된 것으로 보인다.

**다음 실험에서 바꿀 점**

다음 실험에서는 epoch 외에 learning rate나 모델 구조를 조정하여 SGD에서 추가적인 성능 개선이 가능한지 확인한다.

---

#### 6.1.3 SGD-3: learning rate 조정


| 항목                    | 값                           |
| --------------------- | --------------------------- |
| Optimizer             | SGD                         |
| Learning rate         | `0.01`                      |
| Epochs                | `40`                        |
| Batch size            | `128`                       |
| Hidden layer          | `784 -> 512 -> 256 -> 10`   |
| BatchNorm             | 사용                         |
| Dropout               | 사용 (`drop_ratio=0.5`)      |


**실험 의도**

이전 실험에서 epoch을 늘렸을 때 loss와 정확도가 개선되었지만, SGD의 학습 속도는 여전히 느린 편이었다. 따라서 learning rate를 `0.001`에서 `0.01`로 높이면 한 번의 업데이트에서 더 크게 이동하여 loss가 더 빠르게 감소하고 최종 정확도도 개선될 수 있다고 가정했다.

**결과**


| 항목            | 결과     |
| ------------- | ------ |
| Train loss    | `0.2165` |
| Test accuracy | `95.94%` |
| 총 파라미터 수      | `537,354` |


**스크린샷**

- 학습 로그: 
<img src="image/스크린샷 2026-05-25 오후 10.47.27.png" alt="스크린샷" width="700">
- Loss curve: 
<img src="image/스크린샷 2026-05-25 오후 10.47.54.png" alt="스크린샷" width="700">

**분석**

learning rate를 `0.001`에서 `0.01`로 높이자 loss가 `0.5479`에서 `0.2165`까지 크게 감소했고, test accuracy도 `90.33%`에서 `95.94%`로 향상되었다. 초반 epoch부터 loss가 빠르게 줄어들었고 train accuracy와 test accuracy가 각각 `96.25%`, `95.94%`로 비슷하게 유지되어, 이번 설정에서는 learning rate 증가가 수렴 속도와 최종 성능을 모두 개선한 것으로 보인다.

**다음 실험에서 바꿀 점**

SGD 실험 중 가장 좋은 결과가 나왔으므로, 다음에는 optimizer를 Adam으로 바꿔 같은 구조에서 수렴 속도와 정확도를 비교한다.

### 6.2 Adam 실험

#### 6.2.1 Adam-1: SGD 최적 설정 기반 optimizer 변경


| 항목                    | 값                           |
| --------------------- | --------------------------- |
| Optimizer             | Adam                        |
| Learning rate         | `0.001`                     |
| Epochs                | `20`                        |
| Batch size            | `128`                       |
| Hidden layer          | `784 -> 512 -> 256 -> 10`   |
| BatchNorm             | 사용                         |
| Dropout               | 사용 (`drop_ratio=0.5`)      |


**실험 의도**

SGD에서 가장 괜찮았던 구조를 유지하고 optimizer만 Adam으로 바꿔 수렴 속도와 정확도를 비교한다. Adam은 gradient의 이동평균과 제곱 이동평균을 함께 사용하므로, SGD보다 더 빠르게 loss가 감소하고 높은 정확도에 도달할 수 있을 것으로 예상했다.

**결과**


| 항목            | 결과     |
| ------------- | ------ |
| Train loss    | `0.0591` |
| Test accuracy | `98.43%` |
| 총 파라미터 수      | `537,354` |


**스크린샷**

- 학습 로그: 
<img src="image/스크린샷 2026-05-25 오후 10.51.43.png" alt="스크린샷" width="700">
- Loss curve: 
<img src="image/스크린샷 2026-05-25 오후 10.52.25.png" alt="스크린샷" width="700">

**분석**

Adam으로 변경하자 첫 epoch부터 test accuracy가 `96.04%`까지 올라갔고, 20 epoch만으로 최종 test accuracy `98.43%`를 달성했다. SGD 최적 실험의 `95.94%`보다 높은 정확도를 더 적은 epoch에서 얻었으므로, Adam이 이 구조에서는 수렴 속도와 최종 성능 모두에서 더 유리했다.

**다음 실험에서 바꿀 점**

다음 실험에서는 Adam의 learning rate를 조정하여 현재 설정보다 더 안정적이거나 높은 정확도를 얻을 수 있는지 확인한다.

---

#### 6.2.2 Adam-2: epoch 조정


| 항목                    | 값                           |
| --------------------- | --------------------------- |
| Optimizer             | Adam                        |
| Learning rate         | `0.001`                     |
| Epochs                | `40`                        |
| Batch size            | `128`                       |
| Hidden layer          | `784 -> 512 -> 256 -> 10`   |
| BatchNorm             | 사용                         |
| Dropout               | 사용 (`drop_ratio=0.5`)      |


**실험 의도**

Adam-1에서 이미 높은 정확도를 얻었기 때문에, 같은 learning rate에서 epoch을 `20`에서 `40`으로 늘렸을 때 loss와 정확도가 추가로 개선되는지 확인한다. epoch을 늘리면 train loss는 더 감소하겠지만, test accuracy가 함께 좋아지는지 또는 과적합 경향이 생기는지를 비교한다.

**결과**


| 항목            | 결과     |
| ------------- | ------ |
| Train loss    | `0.0409` |
| Test accuracy | `98.34%` |
| 총 파라미터 수      | `537,354` |


**스크린샷**

- 학습 로그: 
<img src="image/스크린샷 2026-05-25 오후 10.55.36.png" alt="스크린샷" width="700">
- Loss curve: 
<img src="image/스크린샷 2026-05-25 오후 10.56.16.png" alt="스크린샷" width="700">

**분석**

epoch을 `40`으로 늘리자 train loss는 `0.0591`에서 `0.0409`까지 더 감소했다. 하지만 test accuracy는 `98.43%`에서 `98.34%`로 약간 낮아졌고, train accuracy는 `99.90%`까지 올라간 반면 test accuracy는 `98.34%`에 머물렀다. 따라서 Adam에서는 epoch을 늘리는 것이 학습 데이터에는 더 잘 맞게 만들었지만, 일반화 성능 개선에는 크게 도움이 되지 않았고 약한 과적합 경향이 보인다.

**다음 실험에서 바꿀 점**

다음 실험에서는 epoch을 무작정 늘리기보다 learning rate나 dropout 비율을 조정하여 test accuracy가 더 안정적으로 개선되는지 확인한다.

---

#### 6.2.3 Adam-3: learning rate 조정


| 항목                    | 값                           |
| --------------------- | --------------------------- |
| Optimizer             | Adam                        |
| Learning rate         | `0.0005`                    |
| Epochs                | `40`                        |
| Batch size            | `128`                       |
| Hidden layer          | `784 -> 512 -> 256 -> 10`   |
| BatchNorm             | 사용                         |
| Dropout               | 사용 (`drop_ratio=0.5`)      |


**실험 의도**

이전 Adam 실험에서 epoch을 늘렸을 때 train loss는 감소했지만 test accuracy는 크게 개선되지 않았다. 따라서 learning rate를 `0.001`에서 `0.0005`로 낮춰 업데이트 폭을 줄이고, 40 epoch 동안 더 안정적으로 수렴시켜 정확도가 개선되는지 확인한다.

**결과**


| 항목            | 결과     |
| ------------- | ------ |
| Train loss    | `0.0390` |
| Test accuracy | `98.51%` |
| 총 파라미터 수      | `537,354` |


**스크린샷**

- 학습 로그: 
<img src="image/스크린샷 2026-05-25 오후 11.03.24.png" alt="스크린샷" width="700">
- Loss curve: 
<img src="image/스크린샷 2026-05-25 오후 11.03.44.png" alt="스크린샷" width="700">

**분석**

learning rate를 `0.0005`로 낮추자 train loss는 `0.0390`, test accuracy는 `98.51%`가 나왔다. Adam-1의 `98.43%`, Adam-2의 `98.34%`보다 test accuracy가 높아졌고, train accuracy `99.89%`와 test accuracy `98.51%`의 차이도 과도하지 않았다. 따라서 현재까지의 실험 중 가장 좋은 정확도를 보인 설정이다.

**다음 실험에서 바꿀 점**

다음 실험에서는 같은 learning rate에서 epoch을 조금 더 늘려 test accuracy가 추가로 개선되는지 확인한다.

---

#### 6.2.4 Adam-4: epoch 추가 조정

| 항목                    | 값                           |
| --------------------- | --------------------------- |
| Optimizer             | Adam                        |
| Learning rate         | `0.0005`                    |
| Epochs                | `60`                        |
| Batch size            | `128`                       |
| Hidden layer          | `784 -> 512 -> 256 -> 10`   |
| BatchNorm             | 사용                         |
| Dropout               | 사용 (`drop_ratio=0.5`)      |

**실험 의도**

Adam-3에서 learning rate를 낮췄을 때 test accuracy가 가장 높았고 loss도 안정적으로 감소했다. 따라서 같은 learning rate에서 epoch을 `40`에서 `60`으로 늘리면 모델이 더 충분히 학습되어 test accuracy가 추가로 개선되는지 확인한다.

**결과**

| 항목            | 결과     |
| ------------- | ------ |
| Train loss    | `0.0309` |
| Test accuracy | `98.50%` |
| 총 파라미터 수      | `537,354` |

**스크린샷**

- 학습 로그: 
<img src="image/스크린샷 2026-05-25 오후 11.07.20.png" alt="스크린샷" width="700">
- Loss curve: 
<img src="image/스크린샷 2026-05-25 오후 11.08.45.png" alt="스크린샷" width="700">

**분석**

epoch을 `60`으로 늘리자 train loss는 `0.0390`에서 `0.0309`까지 더 감소했다. 하지만 test accuracy는 `98.51%`에서 `98.50%`로 거의 같거나 아주 약간 낮아졌고, 중간 epoch에서도 `98.56%` 근처에서 변동했다. 따라서 추가 학습은 학습 데이터 loss를 더 낮추는 데는 효과가 있었지만, 최종 test accuracy 개선으로 이어지지는 않았다.

**다음 실험에서 바꿀 점**

Adam-3과 Adam-4의 test accuracy 차이가 거의 없으므로, 다음 실험에서는 epoch이 아니라 Dropout 비율을 조정하여 정규화 강도가 성능에 미치는 영향을 확인한다.

---

#### 6.2.5 Adam-5: Dropout 비율 조정

| 항목                    | 값                           |
| --------------------- | --------------------------- |
| Optimizer             | Adam                        |
| Learning rate         | `0.0005`                    |
| Epochs                | `40`                        |
| Batch size            | `128`                       |
| Hidden layer          | `784 -> 512 -> 256 -> 10`   |
| BatchNorm             | 사용                         |
| Dropout               | 사용 (`drop_ratio=0.3`)      |

**실험 의도**

기존 `drop_ratio=0.5`는 과적합 방지에는 도움이 되지만, 유용한 특징까지 많이 제거해 학습을 방해했을 가능성이 있다. 따라서 Dropout 비율을 `0.5`에서 `0.3`으로 낮추어 정규화 효과는 유지하면서 더 많은 특징을 활용할 수 있는지 확인한다.

**결과**

| 항목            | 결과     |
| ------------- | ------ |
| Train loss    | `0.0142` |
| Test accuracy | `98.41%` |
| 총 파라미터 수      | `537,354` |

**스크린샷**

- 학습 로그: 
<img src="image/스크린샷 2026-05-25 오후 11.14.10.png" alt="스크린샷" width="700">

- Loss curve: 
<img src="image/스크린샷 2026-05-25 오후 11.14.45.png" alt="스크린샷" width="700">

**분석**

Dropout 비율을 `0.5`에서 `0.3`으로 낮추자 train loss는 `0.0142`까지 크게 감소했고 train accuracy도 `99.98%`까지 올라갔다. 하지만 최종 test accuracy는 `98.41%`로 Adam-3의 `98.51%`보다 낮았다. 중간 epoch에서 `98.57%`까지 올라간 구간은 있었지만 최종 결과 기준으로는 일반화 성능이 개선되지 않았고, Dropout을 약하게 하면서 학습 데이터에 더 강하게 맞는 경향이 나타났다.

**다음 실험에서 바꿀 점**

Dropout 비율을 낮추자 train loss는 크게 감소했지만 test accuracy는 개선되지 않았으므로, 다음 실험에서는 Dropout 비율을 `0.6`으로 높여 정규화 강도를 더 강하게 했을 때 과적합이 줄어드는지 확인한다.

---

#### 6.2.6 Adam-6: Dropout 비율 추가 조정

| 항목                    | 값                           |
| --------------------- | --------------------------- |
| Optimizer             | Adam                        |
| Learning rate         | `0.0005`                    |
| Epochs                | `40`                        |
| Batch size            | `128`                       |
| Hidden layer          | `784 -> 512 -> 256 -> 10`   |
| BatchNorm             | 사용                         |
| Dropout               | 사용 (`drop_ratio=0.6`)      |

**실험 의도**

Dropout 비율을 `0.3`으로 낮췄을 때 train loss는 크게 감소했지만 test accuracy는 개선되지 않아 과적합 경향이 커진 것으로 보였다. 따라서 Dropout 비율을 `0.6`으로 높여 정규화를 더 강하게 적용하면 과적합을 줄이고 test accuracy가 개선되는지 확인한다.

**결과**

| 항목            | 결과     |
| ------------- | ------ |
| Train loss    | `0.0720` |
| Test accuracy | `98.40%` |
| 총 파라미터 수      | `537,354` |

**스크린샷**

- 학습 로그: 
<img src="image/스크린샷 2026-05-25 오후 11.19.14.png" alt="스크린샷" width="700">

- Loss curve: 
<img src="image/스크린샷 2026-05-25 오후 11.19.53.png" alt="스크린샷" width="700">

**분석**

Dropout 비율을 `0.6`으로 높이자 train loss는 `0.0720`으로 Adam-5의 `0.0142`보다 높아졌고, train accuracy도 `99.70%`로 낮아졌다. 이는 정규화가 더 강하게 적용되어 학습 데이터에 과하게 맞는 정도는 줄어든 것으로 볼 수 있다. 하지만 test accuracy는 `98.40%`로 Adam-3의 `98.51%`보다 낮았고 Adam-5의 `98.41%`와도 거의 차이가 없었다. 따라서 `drop_ratio=0.6`은 과적합을 줄이는 방향으로 작용했지만, 최종 일반화 성능을 개선하지는 못했다.

**다음 실험에서 바꿀 점**

Dropout 비율을 `0.6`으로 높였을 때 train loss는 높아졌지만 test accuracy는 개선되지 않았다. 다음 실험에서는 같은 Dropout 비율에서 epoch을 크게 늘려 더 오래 학습했을 때 성능 변화가 있는지 확인한다.

---

#### 6.2.7 Adam-7: 강한 Dropout에서 epoch 추가 조정

| 항목                    | 값                           |
| --------------------- | --------------------------- |
| Optimizer             | Adam                        |
| Learning rate         | `0.0005`                    |
| Epochs                | `100`                       |
| Batch size            | `128`                       |
| Hidden layer          | `784 -> 512 -> 256 -> 10`   |
| BatchNorm             | 사용                         |
| Dropout               | 사용 (`drop_ratio=0.6`)      |

**실험 의도**

Dropout 비율을 `0.6`으로 높였을 때 정규화가 강해져 train loss는 높아졌지만 test accuracy는 개선되지 않았다. 따라서 같은 Dropout 비율에서 epoch을 `40`에서 `100`으로 늘리면 강한 정규화 상태에서도 모델이 충분히 학습되어 test accuracy가 개선되는지 확인한다.

**결과**

| 항목            | 결과     |
| ------------- | ------ |
| Train loss    | `0.0456` |
| Test accuracy | `98.39%` |
| 총 파라미터 수      | `537,354` |

**스크린샷**

- 학습 로그: 
<img src="image/스크린샷 2026-05-25 오후 11.25.29.png" alt="스크린샷" width="700">

- Loss curve: 
<img src="image/스크린샷 2026-05-25 오후 11.25.54.png" alt="스크린샷" width="700">

**분석**

epoch을 `100`으로 늘리자 train loss는 `0.0720`에서 `0.0456`까지 감소했다. 하지만 test accuracy는 `98.40%`에서 `98.39%`로 거의 같거나 아주 약간 낮아졌고, 중간 epoch에서는 `98.60%`까지 올라간 구간도 있었지만 최종 결과로 유지되지는 않았다. 따라서 강한 Dropout 상태에서 학습을 오래 진행해도 최종 일반화 성능은 개선되지 않았다.

**다음 실험에서 바꿀 점**

Dropout 비율과 epoch을 추가로 조정해도 Adam-3의 test accuracy `98.51%`를 안정적으로 넘지 못했다. 다음 실험에서는 Adam-3의 Dropout 비율 `0.5`를 유지하되 epoch만 `60`으로 늘려 성능 변화가 있는지 한 번 더 확인한다.

---

#### 6.2.8 Adam-8: 기본 Dropout에서 epoch 추가 조정

| 항목                    | 값                           |
| --------------------- | --------------------------- |
| Optimizer             | Adam                        |
| Learning rate         | `0.0005`                    |
| Epochs                | `60`                        |
| Batch size            | `128`                       |
| Hidden layer          | `784 -> 512 -> 256 -> 10`   |
| BatchNorm             | 사용                         |
| Dropout               | 사용 (`drop_ratio=0.5`)      |

**실험 의도**

Dropout 비율을 `0.3`, `0.6`으로 바꾼 실험에서는 Adam-3보다 좋은 최종 test accuracy를 얻지 못했다. 따라서 Adam-3의 `drop_ratio=0.5`를 유지하고 epoch만 `40`에서 `60`으로 늘려, 기본 Dropout 비율에서 추가 학습이 성능 개선으로 이어지는지 확인한다.

**결과**

| 항목            | 결과     |
| ------------- | ------ |
| Train loss    | `0.0337` |
| Test accuracy | `98.51%` |
| 총 파라미터 수      | `537,354` |

**스크린샷**

- 학습 로그: 
<img src="image/스크린샷 2026-05-25 오후 11.33.28.png" alt="스크린샷" width="700">

- Loss curve: 
<img src="image/스크린샷 2026-05-25 오후 11.33.51.png" alt="스크린샷" width="700">

**분석**

epoch을 `60`으로 늘리자 train loss는 Adam-3의 `0.0390`에서 `0.0337`로 더 감소했다. test accuracy는 `98.51%`로 Adam-3과 같았고, 중간 epoch에서는 `98.59%`까지 올라간 구간도 있었지만 최종 결과로는 추가 개선이 유지되지 않았다. 따라서 기본 Dropout 비율에서 epoch을 더 늘리는 것은 학습 loss 감소에는 도움이 되었지만, 최종 test accuracy를 안정적으로 높이지는 못했다.

**다음 실험에서 바꿀 점**

Adam-8도 Adam-3과 같은 test accuracy를 보였으므로, 더 짧은 epoch으로 같은 최종 정확도를 얻은 Adam-3 설정을 최종 모델 후보로 선택한다.

## 7. 실험 요약 비교

### 7.1 전체 실험 결과 요약


| 실험     | Optimizer | 주요 변경점                     | Train loss | Test accuracy | 분석 요약 |
| ------ | --------- | -------------------------- | ---------- | ------------- | ------ |
| SGD-1  | SGD       | 기본 설정                      | `0.7770`   | `86.87%`      | 학습은 정상 진행되었지만 정확도는 낮았다. |
| SGD-2  | SGD       | epoch 조정                    | `0.5479`   | `90.33%`      | epoch 증가로 loss와 정확도가 함께 개선되었다. |
| SGD-3  | SGD       | learning rate 변경            | `0.2165`   | `95.94%`      | learning rate 증가가 SGD의 수렴 속도와 성능을 크게 개선했다. |
| Adam-1 | Adam      | optimizer 변경                | `0.0591`   | `98.43%`      | SGD보다 적은 epoch에서 더 높은 정확도에 도달했다. |
| Adam-2 | Adam      | epoch 조정                    | `0.0409`   | `98.34%`      | train loss는 감소했지만 test accuracy는 오히려 낮아졌다. |
| Adam-3 | Adam      | learning rate 변경            | `0.0390`   | `98.51%`      | 낮춘 learning rate가 가장 안정적인 최종 성능을 보였다. |
| Adam-4 | Adam      | epoch 추가 조정                | `0.0309`   | `98.50%`      | 추가 학습으로 loss는 줄었지만 정확도 개선은 거의 없었다. |
| Adam-5 | Adam      | Dropout 비율 감소              | `0.0142`   | `98.41%`      | Dropout을 약하게 하자 train loss는 크게 줄었지만 일반화 성능은 개선되지 않았다. |
| Adam-6 | Adam      | Dropout 비율 증가              | `0.0720`   | `98.40%`      | 강한 Dropout은 과적합을 줄였지만 최종 정확도는 높이지 못했다. |
| Adam-7 | Adam      | 강한 Dropout에서 epoch 추가 조정 | `0.0456`   | `98.39%`      | 강한 Dropout 상태에서 오래 학습해도 test accuracy는 개선되지 않았다. |
| Adam-8 | Adam      | 기본 Dropout에서 epoch 추가 조정 | `0.0337`   | `98.51%`      | Adam-3과 같은 정확도를 보였지만 더 많은 epoch이 필요했다. |


### 7.2 SGD와 Adam 비교


| 비교 항목    | SGD                                 | Adam                                 |
| -------- | ----------------------------------- | ------------------------------------ |
| 업데이트 방식  | gradient에 learning rate를 곱해 직접 업데이트 | momentum과 adaptive learning rate를 사용 |
| 수렴 속도    | 느린 편이다. epoch과 learning rate를 조정해야 성능이 올라갔다. | 빠른 편이다. 20 epoch만으로도 98%대 정확도에 도달했다. |
| 최종 정확도   | 최고 `95.94%`                       | 최고 `98.51%`                        |
| loss 안정성 | learning rate가 낮을 때는 안정적이지만 수렴이 느렸다. | 전반적으로 빠르고 안정적으로 감소했지만 epoch을 늘리면 약한 과적합이 보였다. |
| 장점       | 구조가 단순하고 업데이트 과정을 이해하기 쉽다.       | 파라미터별 업데이트 폭을 조절해 빠르게 수렴한다.       |
| 단점       | 적절한 learning rate를 찾지 않으면 성능이 낮다.  | 너무 오래 학습하면 train loss만 줄고 test accuracy는 개선되지 않을 수 있다. |


## 8. 최종 모델


| 항목                    | 최종 설정            |
| --------------------- | ---------------- |
| Optimizer             | Adam             |
| Learning rate         | `0.0005`         |
| Epochs                | `40`             |
| Batch size            | `128`            |
| Hidden layer          | `784 -> 512 -> 256 -> 10` |
| BatchNorm             | 사용              |
| Dropout               | 사용 (`drop_ratio=0.5`) |
| Test accuracy         | `98.51%`         |
| 총 파라미터 수              | `537,354`        |
| 목표 정확도 달성 여부          | 달성              |


최종 모델로 선택한 이유:

Adam-3은 test accuracy `98.51%`로 전체 실험 중 가장 높은 최종 정확도를 기록했다. Adam-8도 같은 정확도를 보였지만 epoch이 `60`으로 더 길었기 때문에, 같은 성능을 더 적은 학습 횟수로 얻은 Adam-3이 더 효율적인 최종 모델이라고 판단했다. 또한 Dropout 비율을 `0.3`이나 `0.6`으로 조정한 실험에서는 test accuracy가 오히려 낮아졌으므로, 기본 설정인 `drop_ratio=0.5`가 이 모델에서는 가장 적절했다.

## 9. 회고

### 9.1 잘 된 점

- ReLU, Affine, Softmax, BatchNorm, Dropout의 forward/backward 흐름을 직접 구현하면서 각 레이어가 학습 과정에서 어떤 역할을 하는지 확인할 수 있었다.
- 단위 테스트를 단계별로 실행하면서 구현 오류를 빠르게 찾고, 수정 후 다시 검증하는 흐름을 만들 수 있었다.

### 9.2 어려웠던 점

- BatchNorm backward는 중간 변수와 gradient 흐름이 많아서 계산 과정을 이해하고 구현하는 데 시간이 걸렸다.
- mini-batch 학습에서 마지막 batch 크기처럼 작은 예외 상황이 있어, 고정된 `batch_size`를 그대로 쓰면 오류가 날 수 있다는 점을 디버깅 과정에서 확인했다.

### 9.3 개선 과정에서 배운 점

- optimizer를 SGD에서 Adam으로 바꾸는 것만으로도 수렴 속도와 최종 정확도가 크게 달라질 수 있음을 확인했다.
- learning rate를 적절히 조정하면 같은 모델 구조에서도 더 안정적으로 높은 정확도에 도달할 수 있었다.
- Dropout 비율을 낮추면 train loss는 크게 줄어들 수 있지만, test accuracy가 반드시 좋아지는 것은 아니며 일반화 성능을 함께 봐야 한다.

### 9.4 최종 정리

이번 과제를 통해 NumPy만으로 신경망의 핵심 구성 요소를 구현하면서, 모델 학습이 단순한 함수 호출이 아니라 forward, loss, backward, update가 연결된 과정임을 확인했다.

최종적으로 Adam optimizer, learning rate `0.0005`, epoch `40`, Dropout `0.5` 설정에서 test accuracy `98.51%`를 달성했다. 목표 정확도 95%와 권장 목표 97%를 모두 넘겼으며, 가장 효과적이었던 개선 방법은 SGD에서 Adam으로 optimizer를 변경하고 learning rate를 낮춰 안정적으로 수렴시키는 것이었다.
