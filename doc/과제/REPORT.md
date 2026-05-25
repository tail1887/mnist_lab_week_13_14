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
/Users/tail1/Desktop/krafton-jungle/AI/mnist_lab_week_13_14/doc/과제/image/스크린샷 2026-05-25 오후 1.54.11.png

- 테스트 결과 스크린샷: 
/Users/tail1/Desktop/krafton-jungle/AI/mnist_lab_week_13_14/doc/과제/image/스크린샷 2026-05-25 오후 1.55.55.png

- 수정 내용 및 확인한 점: 
레루 함수에서 들어오는 x는 affine forward함수의 리턴값이다.
역전파에서 마스크값이 0인 a와 연결된 모든 노드는 막힌다
레루함수를 적용하는김에 마스크를 만들고 이 마스크는 역전파에서 신호가 없는 노드를 표현하기 위해서 활용한다.

### 3.2 Step 2: Softmax

- 구현 대상: `Softmax.forward`, `Softmax.backward`
- 수정 파일: `src/activations.py`
- 테스트 명령: `pytest tests/test_softmax.py -v`
- 코드 스크린샷: 
/Users/tail1/Desktop/krafton-jungle/AI/mnist_lab_week_13_14/doc/과제/image/스크린샷 2026-05-25 오후 2.01.29.png

- 테스트 결과 스크린샷: 
/Users/tail1/Desktop/krafton-jungle/AI/mnist_lab_week_13_14/doc/과제/image/스크린샷 2026-05-25 오후 2.03.04.png

- 수정 내용 및 확인한 점: 
np.max(x)를 그냥 사용해도 테스트가 통과하는데 이부분은 입력이 1차원 벡터면 상관없지만 그 외에는 문제가 발생하게 된다. 
예: 사진이 여러장일 때, 각각의 최댓값이 아닌 하나의 최댓값을 공유하게됨

### 3.3 Step 3: Affine

- 구현 대상: `Affine.forward`, `Affine.backward`
- 수정 파일: `src/layers.py`
- 테스트 명령: `pytest tests/test_affine.py -v`
- 코드 스크린샷: `TODO` 이미지 첨부
/Users/tail1/Desktop/krafton-jungle/AI/mnist_lab_week_13_14/doc/과제/image/스크린샷 2026-05-25 오후 2.47.26.png

- 테스트 결과 스크린샷: 
/Users/tail1/Desktop/krafton-jungle/AI/mnist_lab_week_13_14/doc/과제/image/스크린샷 2026-05-25 오후 2.49.33.png

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
/Users/tail1/Desktop/krafton-jungle/AI/mnist_lab_week_13_14/doc/과제/image/스크린샷 2026-05-25 오후 3.46.35.png

- 테스트 결과 스크린샷:
/Users/tail1/Desktop/krafton-jungle/AI/mnist_lab_week_13_14/doc/과제/image/스크린샷 2026-05-25 오후 3.47.39.png

- 수정 내용 및 확인한 점:
교차 엔트로피 오차는 소프트 맥스 함수와 환상의 짝궁이다. 역전파 과정에서 교차 엔트로피의 미분값이랑 소프트맥스의 미분값이랑 곱하면 항들이 정리되어 식이 깔끔해진다. 

### 3.5 Step 5: SGD

- 구현 대상: `SGD.update`
- 수정 파일: `src/optimizers.py`
- 테스트 명령: `pytest tests/test_sgd.py -v`
- 코드 스크린샷: `TODO` 이미지 첨부
- 테스트 결과 스크린샷: `TODO` 이미지 첨부
- 수정 내용 및 확인한 점: `TODO`

### 3.6 Step 6: Adam

- 구현 대상: `Adam.update`
- 수정 파일: `src/optimizers.py`
- 테스트 명령: `pytest tests/test_adam.py -v`
- 코드 스크린샷: `TODO` 이미지 첨부
- 테스트 결과 스크린샷: `TODO` 이미지 첨부
- 수정 내용 및 확인한 점: `TODO`

### 3.7 Step 7: NeuralNetwork

- 구현 대상: `NeuralNetwork`
- 수정 파일: `src/network.py`
- 테스트 명령: `pytest tests/test_neural_network.py -v`
- 코드 스크린샷: `TODO` 이미지 첨부
- 테스트 결과 스크린샷: `TODO` 이미지 첨부
- 수정 내용 및 확인한 점: `TODO`

### 3.8 Step 8: BatchNorm

- 구현 대상: `BatchNorm.forward`, `BatchNorm.backward`
- 수정 파일: `src/layers.py`
- 테스트 명령: `pytest tests/test_batchnorm.py -v`
- 코드 스크린샷: `TODO` 이미지 첨부
- 테스트 결과 스크린샷: `TODO` 이미지 첨부
- 수정 내용 및 확인한 점: `TODO`

### 3.9 Step 9: Dropout

- 구현 대상: `Dropout.forward`, `Dropout.backward`
- 수정 파일: `src/layers.py`
- 테스트 명령: `pytest tests/test_dropout.py -v`
- 코드 스크린샷: `TODO` 이미지 첨부
- 테스트 결과 스크린샷: `TODO` 이미지 첨부
- 수정 내용 및 확인한 점: `TODO`

### 3.10 Step 10: Train

- 구현 대상: `train`
- 수정 파일: `src/training.py`
- 테스트 명령: `pytest tests/test_training.py -v`
- 코드 스크린샷: `TODO` 이미지 첨부
- 테스트 결과 스크린샷: `TODO` 이미지 첨부
- 수정 내용 및 확인한 점: `TODO`

전체 테스트:

```bash
pytest tests/ -v
```

**테스트 과정에서 발견한 문제와 수정 내용**

- `TODO`: 예) `Affine.forward`에서 `self.w`를 `self.W`로 수정했다.
- `TODO`: 예) BatchNorm의 train/test 동작 구분을 수정했다.
- `TODO`: 예) Dropout의 mask 저장 위치를 수정했다.


## 4. 실험 환경


| 항목             | 내용                                    |
| -------------- | ------------------------------------- |
| Python version | 3.11                                  |
| 사용 라이브러리       | NumPy, math, random, time, matplotlib |
| 실행 환경          | `TODO`: Google Colab / 로컬 Conda       |
| 하드웨어           | `TODO`: CPU / GPU                     |
| 데이터셋           | MNIST                                 |


## 5. 실험 진행 방식

이번 실험은 한 번에 최종 모델을 만든 것이 아니라, 기본 구현을 검증한 뒤 학습 결과를 분석하고 설정값을 바꾸며 성능을 개선하는 방식으로 진행했다.

실험 흐름:

1. 기본 모델로 학습이 정상적으로 진행되는지 확인한다.
2. `SGD`로 baseline 성능을 측정한다.
3. loss와 accuracy를 보고 문제점을 분석한다.
4. learning rate, epoch, hidden layer, dropout 등을 조정한다.
5. `Adam`으로 optimizer를 변경해 수렴 속도와 정확도를 비교한다.
6. 가장 좋은 설정을 최종 모델로 선택한다.

## 6. 실험 기록

### 6.1 SGD 실험

#### 6.1.1 SGD-1: 기본 설정 실험


| 항목                    | 값                           |
| --------------------- | --------------------------- |
| Optimizer             | SGD                         |
| Learning rate         | `TODO`                      |
| Epochs                | `TODO`                      |
| Batch size            | `TODO`                      |
| Hidden layer          | `TODO`                      |
| BatchNorm             | `TODO`: 사용 / 미사용            |
| Dropout               | `TODO`                      |
| Weight initialization | `TODO`: He / Xavier / 직접 설정 |
| 학습 시간                 | `TODO`                      |


**실험 의도**

`TODO`: 먼저 SGD로 기본 학습이 정상적으로 진행되는지 확인한다.

**결과**


| 항목            | 결과     |
| ------------- | ------ |
| Train loss    | `TODO` |
| Test accuracy | `TODO` |
| 총 파라미터 수      | `TODO` |


**스크린샷**

- 학습 로그: `TODO` 이미지 첨부
- Loss curve: `TODO` 이미지 첨부
- Test accuracy: `TODO` 이미지 첨부

**분석**

`TODO`: loss가 감소했는지, 정확도가 충분했는지, 과소적합/과적합이 보였는지 작성한다.

**다음 실험에서 바꿀 점**

`TODO`: 예) learning rate를 낮춘다 / epoch를 늘린다 / hidden layer 크기를 조정한다.

---

#### 6.1.2 SGD-2: learning rate 조정


| 항목                    | 값                           |
| --------------------- | --------------------------- |
| Optimizer             | SGD                         |
| Learning rate         | `TODO`                      |
| Epochs                | `TODO`                      |
| Batch size            | `TODO`                      |
| Hidden layer          | `TODO`                      |
| BatchNorm             | `TODO`: 사용 / 미사용            |
| Dropout               | `TODO`                      |
| Weight initialization | `TODO`: He / Xavier / 직접 설정 |
| 학습 시간                 | `TODO`                      |


**실험 의도**

`TODO`: 이전 실험의 결과를 바탕으로 learning rate를 조정해 loss 감소와 정확도 변화를 확인한다.

**결과**


| 항목            | 결과     |
| ------------- | ------ |
| Train loss    | `TODO` |
| Test accuracy | `TODO` |
| 총 파라미터 수      | `TODO` |


**스크린샷**

- 학습 로그: `TODO` 이미지 첨부
- Loss curve: `TODO` 이미지 첨부
- Test accuracy: `TODO` 이미지 첨부

**분석**

`TODO`

**다음 실험에서 바꿀 점**

`TODO`

---

#### 6.1.3 SGD-3: 모델 구조 또는 epoch 조정


| 항목                    | 값                           |
| --------------------- | --------------------------- |
| Optimizer             | SGD                         |
| Learning rate         | `TODO`                      |
| Epochs                | `TODO`                      |
| Batch size            | `TODO`                      |
| Hidden layer          | `TODO`                      |
| BatchNorm             | `TODO`: 사용 / 미사용            |
| Dropout               | `TODO`                      |
| Weight initialization | `TODO`: He / Xavier / 직접 설정 |
| 학습 시간                 | `TODO`                      |


**실험 의도**

`TODO`: hidden layer, epoch, dropout 등 모델 구조 또는 학습 조건을 바꿔 성능 개선 여부를 확인한다.

**결과**


| 항목            | 결과     |
| ------------- | ------ |
| Train loss    | `TODO` |
| Test accuracy | `TODO` |
| 총 파라미터 수      | `TODO` |


**스크린샷**

- 학습 로그: `TODO` 이미지 첨부
- Loss curve: `TODO` 이미지 첨부
- Test accuracy: `TODO` 이미지 첨부

**분석**

`TODO`

**다음 실험에서 바꿀 점**

`TODO`

### 6.2 Adam 실험

#### 6.2.1 Adam-1: SGD 최적 설정 기반 optimizer 변경


| 항목                    | 값                           |
| --------------------- | --------------------------- |
| Optimizer             | Adam                        |
| Learning rate         | `TODO`                      |
| Epochs                | `TODO`                      |
| Batch size            | `TODO`                      |
| Hidden layer          | `TODO`                      |
| BatchNorm             | `TODO`: 사용 / 미사용            |
| Dropout               | `TODO`                      |
| Weight initialization | `TODO`: He / Xavier / 직접 설정 |
| 학습 시간                 | `TODO`                      |


**실험 의도**

`TODO`: SGD에서 가장 괜찮았던 구조를 유지하고 optimizer만 Adam으로 바꿔 수렴 속도와 정확도를 비교한다.

**결과**


| 항목            | 결과     |
| ------------- | ------ |
| Train loss    | `TODO` |
| Test accuracy | `TODO` |
| 총 파라미터 수      | `TODO` |


**스크린샷**

- 학습 로그: `TODO` 이미지 첨부
- Loss curve: `TODO` 이미지 첨부
- Test accuracy: `TODO` 이미지 첨부

**분석**

`TODO`

**다음 실험에서 바꿀 점**

`TODO`

---

#### 6.2.2 Adam-2: learning rate 조정


| 항목                    | 값                           |
| --------------------- | --------------------------- |
| Optimizer             | Adam                        |
| Learning rate         | `TODO`                      |
| Epochs                | `TODO`                      |
| Batch size            | `TODO`                      |
| Hidden layer          | `TODO`                      |
| BatchNorm             | `TODO`: 사용 / 미사용            |
| Dropout               | `TODO`                      |
| Weight initialization | `TODO`: He / Xavier / 직접 설정 |
| 학습 시간                 | `TODO`                      |


**실험 의도**

`TODO`: Adam에서 learning rate를 조정해 수렴 안정성과 최종 정확도 변화를 확인한다.

**결과**


| 항목            | 결과     |
| ------------- | ------ |
| Train loss    | `TODO` |
| Test accuracy | `TODO` |
| 총 파라미터 수      | `TODO` |


**스크린샷**

- 학습 로그: `TODO` 이미지 첨부
- Loss curve: `TODO` 이미지 첨부
- Test accuracy: `TODO` 이미지 첨부

**분석**

`TODO`

**다음 실험에서 바꿀 점**

`TODO`

---

#### 6.2.3 Adam-3: 최종 성능 개선 실험


| 항목                    | 값                           |
| --------------------- | --------------------------- |
| Optimizer             | Adam                        |
| Learning rate         | `TODO`                      |
| Epochs                | `TODO`                      |
| Batch size            | `TODO`                      |
| Hidden layer          | `TODO`                      |
| BatchNorm             | `TODO`: 사용 / 미사용            |
| Dropout               | `TODO`                      |
| Weight initialization | `TODO`: He / Xavier / 직접 설정 |
| 학습 시간                 | `TODO`                      |


**실험 의도**

`TODO`: 이전 Adam 실험 결과를 바탕으로 최종 정확도를 높이기 위한 설정을 적용한다.

**결과**


| 항목            | 결과     |
| ------------- | ------ |
| Train loss    | `TODO` |
| Test accuracy | `TODO` |
| 총 파라미터 수      | `TODO` |


**스크린샷**

- 학습 로그: `TODO` 이미지 첨부
- Loss curve: `TODO` 이미지 첨부
- Test accuracy: `TODO` 이미지 첨부

**분석**

`TODO`

**다음 실험에서 바꿀 점**

`TODO`: 최종 실험인 경우, 최종 모델로 선택한 이유를 작성한다.

## 7. 실험 요약 비교

### 7.1 전체 실험 결과 요약


| 실험     | Optimizer | 주요 변경점           | Train loss | Test accuracy | 분석 요약  |
| ------ | --------- | ---------------- | ---------- | ------------- | ------ |
| SGD-1  | SGD       | 기본 설정            | `TODO`     | `TODO`        | `TODO` |
| SGD-2  | SGD       | learning rate 변경 | `TODO`     | `TODO`        | `TODO` |
| SGD-3  | SGD       | 구조 또는 epoch 변경   | `TODO`     | `TODO`        | `TODO` |
| Adam-1 | Adam      | optimizer 변경     | `TODO`     | `TODO`        | `TODO` |
| Adam-2 | Adam      | learning rate 변경 | `TODO`     | `TODO`        | `TODO` |
| Adam-3 | Adam      | 최종 조정            | `TODO`     | `TODO`        | `TODO` |


### 7.2 SGD와 Adam 비교


| 비교 항목    | SGD                                 | Adam                                 |
| -------- | ----------------------------------- | ------------------------------------ |
| 업데이트 방식  | gradient에 learning rate를 곱해 직접 업데이트 | momentum과 adaptive learning rate를 사용 |
| 수렴 속도    | `TODO`                              | `TODO`                               |
| 최종 정확도   | `TODO`                              | `TODO`                               |
| loss 안정성 | `TODO`                              | `TODO`                               |
| 장점       | `TODO`                              | `TODO`                               |
| 단점       | `TODO`                              | `TODO`                               |


## 8. 최종 모델


| 항목                    | 최종 설정            |
| --------------------- | ---------------- |
| Optimizer             | `TODO`           |
| Learning rate         | `TODO`           |
| Epochs                | `TODO`           |
| Batch size            | `TODO`           |
| Hidden layer          | `TODO`           |
| BatchNorm             | `TODO`           |
| Dropout               | `TODO`           |
| Weight initialization | `TODO`           |
| Test accuracy         | `TODO`           |
| 총 파라미터 수              | `TODO`           |
| 목표 정확도 달성 여부          | `TODO`: 달성 / 미달성 |


최종 모델로 선택한 이유:

`TODO`: 여러 실험 중 해당 설정이 가장 적절했던 이유를 작성한다.

## 9. 회고

### 9.1 잘 된 점

- `TODO`: 예) ReLU, Affine, Softmax 등 각 레이어의 forward/backward 흐름을 이해할 수 있었다.
- `TODO`: 예) 테스트를 단계별로 통과시키며 구현 오류를 빠르게 확인할 수 있었다.

### 9.2 어려웠던 점

- `TODO`: 예) BatchNorm backward 계산 과정이 복잡했다.
- `TODO`: 예) shape 불일치로 인한 오류를 디버깅하는 데 시간이 걸렸다.

### 9.3 개선 과정에서 배운 점

- `TODO`: 예) optimizer 변경만으로도 수렴 속도와 정확도가 달라질 수 있음을 확인했다.
- `TODO`: 예) learning rate가 너무 크거나 작으면 loss 감소가 불안정하거나 느려질 수 있음을 확인했다.
- `TODO`: 예) Dropout과 BatchNorm은 학습 안정성과 일반화 성능에 영향을 준다.

### 9.4 최종 정리

이번 과제를 통해 NumPy만으로 신경망의 핵심 구성 요소를 구현하면서, 모델 학습이 단순한 함수 호출이 아니라 forward, loss, backward, update가 연결된 과정임을 확인했다.

최종적으로 `TODO`: 목표 정확도 달성 여부와 가장 효과적이었던 개선 방법을 작성한다.