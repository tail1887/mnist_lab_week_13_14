# 테스트별 구현 가이드

이 문서는 `tests/` 아래 각 테스트를 통과하기 위해 `src/`의 어느 부분을 구현해야 하는지 정리한 가이드입니다.  
권장 순서는 테스트 파일에 적힌 Step 번호를 따릅니다.

## 빠른 실행법

프로젝트 루트에서 Conda 환경을 활성화한 뒤 실행합니다.

```bash
conda activate mnist-nn
pytest tests/test_relu.py -v
```

특정 테스트만 실행하려면 `-k`를 사용할 수 있습니다.

```bash
pytest tests/test_relu.py -v -k "backward"
```

전체 테스트는 다음 명령으로 실행합니다.

```bash
pytest tests/ -v
```

## Step 1. ReLU

- 테스트 파일: `tests/test_relu.py`
- 구현 파일: `src/activations.py`
- 구현 대상:
  - `ReLU.forward`
  - `ReLU.backward`

`ReLU.forward`는 입력 `x`에서 양수는 그대로 두고, 0 이하 값은 0으로 만들어야 합니다.  
역전파에서 사용할 수 있도록 `x > 0` 위치를 `self.mask`에 저장합니다.

`ReLU.backward`는 `forward` 때 양수였던 위치로만 gradient를 통과시켜야 합니다.  
0 이하였던 위치의 gradient는 0이어야 합니다.

실행:

```bash
pytest tests/test_relu.py -v
```

## Step 2. Softmax

- 테스트 파일: `tests/test_softmax.py`
- 구현 파일: `src/activations.py`
- 구현 대상:
  - `Softmax.forward`
  - `Softmax.backward`

`Softmax.forward`는 각 샘플마다 클래스 확률의 합이 1이 되도록 로짓을 확률로 변환해야 합니다.  
수치 안정성을 위해 `np.max(x, axis=1, keepdims=True)`를 빼고 `np.exp`를 적용합니다.

`Softmax.backward`는 이 템플릿에서는 받은 `dout`을 그대로 반환하면 됩니다.  
Softmax와 Cross Entropy를 합친 gradient는 `train` 함수에서 직접 만들도록 설계되어 있습니다.

실행:

```bash
pytest tests/test_softmax.py -v
```

## Step 3. Affine

- 테스트 파일: `tests/test_affine.py`
- 구현 파일: `src/layers.py`
- 구현 대상:
  - `Affine.forward`
  - `Affine.backward`

`Affine.forward`는 완전연결층 계산 `x @ W + b`를 반환해야 합니다.  
역전파에서 사용할 입력 `x`를 인스턴스 변수로 저장해야 합니다.

`Affine.backward`는 다음 값을 계산해야 합니다.

- `self.dW = x.T @ dout`
- `self.db = np.sum(dout, axis=0)`
- `dx = dout @ W.T`

실행:

```bash
pytest tests/test_affine.py -v
```

## Step 4. Cross Entropy Loss

- 테스트 파일: `tests/test_cross_entropy_loss.py`
- 구현 파일: `src/losses.py`
- 구현 대상:
  - `cross_entropy_loss`

`cross_entropy_loss`는 배치 평균 cross entropy를 스칼라로 반환해야 합니다.  
`y_pred`에서 정답 클래스 확률만 골라 `-log` 평균을 계산합니다.

`log(0)`을 피하기 위해 `np.clip`을 사용합니다.

실행:

```bash
pytest tests/test_cross_entropy_loss.py -v
```

## Step 5. SGD

- 테스트 파일: `tests/test_sgd.py`
- 구현 파일: `src/optimizers.py`
- 구현 대상:
  - `SGD.update`

`SGD.update`는 `params` 딕셔너리의 각 파라미터를 gradient 반대 방향으로 갱신해야 합니다.

```python
params[key] -= lr * grads[key]
```

실행:

```bash
pytest tests/test_sgd.py -v
```

## Step 6. Adam

- 테스트 파일: `tests/test_adam.py`
- 구현 파일: `src/optimizers.py`
- 구현 대상:
  - `Adam.update`

`Adam.update`는 각 파라미터에 대해 다음 상태를 관리해야 합니다.

- `self.m`: gradient 이동평균
- `self.v`: gradient 제곱 이동평균
- `self.t`: update step 수

일반 Adam 공식처럼 bias correction을 적용한 `m_hat`, `v_hat`으로 파라미터를 갱신합니다.  
테스트는 파라미터 값이 실제로 바뀌는지를 확인합니다.

실행:

```bash
pytest tests/test_adam.py -v
```

## Step 7. NeuralNetwork 조립

- 테스트 파일: `tests/test_neural_network.py`
- 구현 파일: `src/network.py`
- 구현 대상:
  - `NeuralNetwork.__init__`
  - `NeuralNetwork.forward`
  - `NeuralNetwork.backward`

`NeuralNetwork.__init__`에서는 다음을 준비해야 합니다.

- `self.params`: optimizer가 업데이트할 파라미터 딕셔너리
- `self.grads`: `self.params`와 같은 key를 가진 gradient 딕셔너리
- `self.layers`: 순서를 보장하는 `OrderedDict`
- `self.softmax`: 출력 확률 변환용 `Softmax`

권장 구조는 README와 주석 기준으로 다음과 같습니다.

```text
784 -> 512 -> 256 -> 10
```

은닉층은 보통 다음 순서로 조립합니다.

```text
Affine -> BatchNorm -> ReLU -> Dropout
```

출력층은 마지막 `Affine` 뒤에 `Softmax`를 적용합니다.

`NeuralNetwork.forward`는 `self.layers`를 순서대로 통과시키고 마지막에 `Softmax.forward`를 호출해야 합니다.  
`BatchNorm`과 `Dropout`은 `train` 값을 넘겨 학습 모드와 추론 모드를 구분해야 합니다.

`NeuralNetwork.backward`는 `Softmax.backward` 후 `self.layers`를 역순으로 통과시키며, `Affine`과 `BatchNorm`에서 계산된 gradient를 `self.grads`에 모아야 합니다.

실행:

```bash
pytest tests/test_neural_network.py -v
```

주의: 이 테스트는 `BatchNorm`과 `Dropout`을 켠 모델을 생성합니다.  
따라서 Step 8, Step 9 구현이 아직 없으면 `forward` 또는 `backward`에서 실패할 수 있습니다.

## Step 8. BatchNorm

- 테스트 파일: `tests/test_batchnorm.py`
- 구현 파일: `src/layers.py`
- 구현 대상:
  - `BatchNorm.forward`
  - `BatchNorm.backward`

`BatchNorm.forward`는 입력과 같은 shape의 출력을 반환해야 합니다.

학습 모드에서는 배치 평균과 분산을 사용합니다.

- `mean = np.mean(x, axis=0)`
- `var = np.var(x, axis=0)`
- `x_centered = x - mean`
- `std = np.sqrt(var + eps)`
- `x_norm = x_centered / std`
- `out = gamma * x_norm + beta`

추론 모드에서는 `running_mean`, `running_var`를 사용해야 합니다.  
학습 모드에서는 running 통계도 갱신해야 합니다.

`BatchNorm.backward`는 최소한 다음 gradient를 계산해야 합니다.

- `self.dbeta`
- `self.dgamma`
- `dx`

실행:

```bash
pytest tests/test_batchnorm.py -v
```

## Step 9. Dropout

- 테스트 파일: `tests/test_dropout.py`
- 구현 파일: `src/layers.py`
- 구현 대상:
  - `Dropout.forward`
  - `Dropout.backward`

`Dropout.forward`는 학습 모드에서 입력과 같은 shape의 mask를 만들고 `x * mask`를 반환해야 합니다.  
추론 모드에서는 이 템플릿 주석과 테스트 기준으로 `x * (1 - drop_ratio)`를 반환합니다.

`Dropout.backward`는 `forward`에서 만든 mask를 `dout`에 곱해 반환합니다.

실행:

```bash
pytest tests/test_dropout.py -v
```

## Step 10. Train

- 테스트 파일: `tests/test_training.py`
- 구현 파일: `src/training.py`
- 구현 대상:
  - `train`

`train`은 미니배치 학습 루프를 구현해야 합니다.

흐름은 다음과 같습니다.

```text
epoch마다 데이터 셔플
-> batch 추출
-> model.forward
-> cross_entropy_loss
-> Softmax + CrossEntropy 결합 gradient 생성
-> model.backward
-> optimizer.update(model.params, model.grads)
-> epoch 평균 loss 저장
```

Softmax + CrossEntropy 결합 gradient는 보통 다음 방식으로 만듭니다.

```python
dout = y_pred.copy()
dout[np.arange(batch_size), y_batch] -= 1
dout /= batch_size
```

테스트는 `history`가 리스트이고, epoch 수만큼 손실이 들어 있으며, 손실이 0 이상인지 확인합니다.

실행:

```bash
pytest tests/test_training.py -v
```

## Evaluate

- 테스트 파일: `tests/test_evaluate.py`
- 구현 파일: `src/training.py`
- 구현 대상:
  - 기본 제공된 `evaluate` 함수

`evaluate`는 이미 구현되어 있습니다.  
다만 내부에서 `model.predict`를 호출하므로 `NeuralNetwork`가 구현되어 있어야 테스트가 의미 있게 실행됩니다.

실행:

```bash
pytest tests/test_evaluate.py -v
```

## 추천 구현 순서

테스트 Step 번호는 거의 구현 순서로 볼 수 있지만, 네트워크 조립 테스트는 `BatchNorm`, `Dropout`에 의존합니다.  
실제로는 아래 순서를 추천합니다.

```text
1. ReLU
2. Softmax
3. Affine
4. Cross Entropy Loss
5. SGD
6. Adam
7. BatchNorm
8. Dropout
9. NeuralNetwork
10. Train
11. Evaluate
12. 전체 테스트
```

마지막 전체 확인:

```bash
pytest tests/ -v
```
