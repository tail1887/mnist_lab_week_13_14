# NeuralNetwork 학습 문서

## 1. 핵심 역할

`NeuralNetwork`는 지금까지 구현한 구성 요소들을 하나의 MNIST 분류 모델로 조립한다.

개별 클래스인 `Affine`, `BatchNorm`, `ReLU`, `Dropout`, `Softmax`는 각각 한 가지 역할만 한다. `NeuralNetwork`는 이들을 정해진 순서로 연결하고, forward와 backward가 전체 모델을 통과하도록 관리한다.

현재 모델 구조는 다음과 같다.

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

## 배경: 왜 여러 층을 쌓는가

MNIST 이미지는 28x28 픽셀을 펼친 784차원 벡터다. 단순한 선형 모델은 입력 픽셀에 가중치를 곱해 바로 10개 클래스를 예측한다.

```text
입력 784 -> Affine(10) -> Softmax
```

이 구조는 모든 결정을 하나의 선형 경계로 처리하려고 한다. 하지만 손글씨 숫자는 같은 숫자라도 모양이 다양하고, 서로 다른 숫자도 일부 획이 비슷하다. 그래서 한 번의 선형 변환만으로는 복잡한 패턴을 충분히 표현하기 어렵다.

은닉층을 넣으면 모델은 중간 표현을 학습할 수 있다.

```text
픽셀 -> 획/부분 패턴 -> 숫자 클래스
```

예를 들어 첫 번째 은닉층은 픽셀 조합에서 간단한 획이나 방향성 같은 특징을 만들고, 두 번째 은닉층은 그 특징들을 조합해 더 숫자다운 표현을 만들 수 있다.

단, `Affine`만 여러 번 쌓으면 여전히 하나의 `Affine`과 같다.

```text
y = (xW1 + b1)W2 + b2
  = x(W1W2) + (b1W2 + b2)
```

그래서 중간에 `ReLU` 같은 비선형 활성화 함수가 필요하다. 비선형성이 들어가야 여러 층을 쌓는 의미가 생긴다.

## 전체 학습 흐름

`NeuralNetwork`는 개별 layer를 조립할 뿐 아니라, 학습 루프와도 연결된다.

```text
1. forward: 입력에서 예측 확률을 만든다.
2. loss: 예측 확률과 정답으로 손실을 계산한다.
3. backward: 손실을 줄이기 위한 gradient를 계산한다.
4. update: optimizer가 파라미터를 바꾼다.
```

역전파는 체인 룰을 여러 층에 반복 적용하는 과정이다.

```text
dL/dW1 = dL/dout * dout/d... * ... * dAffine1/dW1
```

`NeuralNetwork.backward()`가 layer를 역순으로 도는 이유가 바로 이 체인 룰 때문이다. 마지막 출력에서 시작한 gradient가 한 층씩 앞쪽으로 전달되어야 한다.

## 가중치 초기화 배경

현재 코드는 가중치를 다음처럼 초기화한다.

```python
np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / input_dim)
```

이는 ReLU 계열 활성화 함수와 잘 맞는 He 초기화 방식이다.

초기 가중치가 너무 크면 forward 값이 층을 지날수록 커지고, gradient도 불안정해질 수 있다. 반대로 너무 작으면 값과 gradient가 점점 작아져 학습이 잘 안 된다.

He 초기화는 ReLU가 대략 절반의 입력을 0으로 만든다는 점을 고려해 분산을 맞추려는 방법이다.

```text
W의 표준편차 ≈ sqrt(2 / fan_in)
```

여기서 `fan_in`은 해당 층으로 들어오는 입력 차원이다. 첫 번째 층에서는 784, 두 번째 층에서는 512가 된다.

## 2. 파라미터 구조

현재 구현 파일은 `src/network.py`다.

```python
self.params = {
    "W1": np.random.randn(784, 512) * np.sqrt(2.0 / 784),
    "b1": np.zeros(512),
    "gamma1": np.ones(512),
    "beta1": np.zeros(512),
    "W2": np.random.randn(512, 256) * np.sqrt(2.0 / 512),
    "b2": np.zeros(256),
    "gamma2": np.ones(256),
    "beta2": np.zeros(256),
    "W3": np.random.randn(256, 10) * np.sqrt(2.0 / 256),
    "b3": np.zeros(10),
}
```

`params`는 학습 가능한 파라미터를 모아 둔 dict다.

- `W1`, `b1`: 첫 번째 `Affine`
- `gamma1`, `beta1`: 첫 번째 `BatchNorm`
- `W2`, `b2`: 두 번째 `Affine`
- `gamma2`, `beta2`: 두 번째 `BatchNorm`
- `W3`, `b3`: 출력층 `Affine`

`self.grads`는 `params`와 같은 key를 가진다.

```python
self.grads = {key: np.zeros_like(value) for key, value in self.params.items()}
```

optimizer는 `params`와 `grads`를 받아 같은 key끼리 연결해 업데이트한다.

## 3. Layer 조립

현재 구현은 `OrderedDict`를 사용한다.

```python
self.layers = OrderedDict()
```

`OrderedDict`를 쓰는 이유는 layer를 넣은 순서대로 forward를 실행하고, 그 반대 순서로 backward를 실행해야 하기 때문이다.

첫 번째 은닉층은 다음 순서로 추가된다.

```python
self.layers["Affine1"] = Affine(self.params["W1"], self.params["b1"])
self.layers["BatchNorm1"] = BatchNorm(self.params["gamma1"], self.params["beta1"])
self.layers["ReLU1"] = ReLU()
self.layers["Dropout1"] = Dropout(dropout_ratio)
```

두 번째 은닉층도 같은 패턴이고, 마지막에는 출력 점수를 만들기 위한 `Affine3`이 들어간다.

`Softmax`는 `self.layers` 안에 넣지 않고 별도로 둔다.

```python
self.softmax = Softmax()
```

## 4. Forward 흐름

현재 forward 구현은 다음과 같다.

```python
for layer in self.layers.values():
    if isinstance(layer, (BatchNorm, Dropout)):
        x = layer.forward(x, train=train)
    else:
        x = layer.forward(x)
return self.softmax.forward(x)
```

흐름은 다음과 같다.

1. `self.layers`에 들어 있는 layer를 순서대로 통과한다.
2. `BatchNorm`과 `Dropout`은 학습 모드 여부가 필요하므로 `train` 값을 함께 넘긴다.
3. 마지막에 `Softmax`를 적용해 클래스별 확률을 반환한다.

`train=True`이면 학습 모드다.

- `BatchNorm`: 현재 배치의 평균과 분산 사용
- `Dropout`: 랜덤 mask 적용

`train=False`이면 추론 모드다.

- `BatchNorm`: running mean/var 사용
- `Dropout`: 확률적 mask 없이 scale만 적용

## 5. Backward 흐름

현재 backward 구현은 다음과 같다.

```python
dout = self.softmax.backward(dout)
for name, layer in reversed(self.layers.items()):
    dout = layer.backward(dout)
    if isinstance(layer, Affine):
        layer_idx = name.replace("Affine", "")
        self.grads[f"W{layer_idx}"] = layer.dW
        self.grads[f"b{layer_idx}"] = layer.db
    elif isinstance(layer, BatchNorm):
        layer_idx = name.replace("BatchNorm", "")
        self.grads[f"gamma{layer_idx}"] = layer.dgamma
        self.grads[f"beta{layer_idx}"] = layer.dbeta
```

역전파는 forward의 반대 순서로 진행된다.

```text
Softmax
<- Affine3
<- Dropout2
<- ReLU2
<- BatchNorm2
<- Affine2
<- Dropout1
<- ReLU1
<- BatchNorm1
<- Affine1
```

각 layer의 backward는 이전 layer로 넘길 gradient를 반환한다. 동시에 `Affine`과 `BatchNorm`은 자신의 파라미터 gradient를 내부에 저장한다.

`NeuralNetwork.backward()`는 이 값을 `self.grads`에 모아 optimizer가 사용할 수 있게 한다.

## 6. Loss와 Predict

`loss()`는 학습 모드로 forward를 실행한 뒤 cross entropy loss를 계산한다.

```python
y_pred = self.forward(x, train=True)
return cross_entropy_loss(y_pred, y)
```

`predict()`는 추론 모드로 forward를 실행한다.

```python
return self.forward(x, train=False)
```

학습과 추론에서 `BatchNorm`, `Dropout`의 동작이 다르므로 이 구분이 중요하다.

## 7. 구현 포인트

가장 중요한 연결 구조는 다음이다.

```text
Layer backward -> self.grads -> Optimizer update -> self.params 변경
```

`Affine` layer는 `self.params["W1"]` 같은 배열 객체를 직접 참조한다. 그래서 optimizer가 `self.params`의 배열을 in-place로 바꾸면 다음 forward에서 바뀐 파라미터가 사용된다.

또한 layer 이름에서 숫자를 뽑아 gradient key를 만드는 구조이므로 이름 규칙이 중요하다.

```python
layer_idx = name.replace("Affine", "")
self.grads[f"W{layer_idx}"] = layer.dW
```

`Affine1`이면 `W1`, `b1`에 저장된다.

## 8. 자주 헷갈리는 점

`Softmax`는 `self.layers`에 들어 있지 않지만 forward와 backward에는 포함된다.

이는 `Softmax`가 학습 파라미터를 갖지 않고, 마지막 확률 변환만 담당하기 때문이다.

또한 `use_batchnorm`, `use_dropout` 옵션이 꺼질 수 있으므로 forward에서는 실제 들어 있는 layer만 순서대로 실행한다. 이 구조 덕분에 모델 구성을 바꿔도 forward/backward 루프 자체는 크게 바뀌지 않는다.

## 9. 테스트 관점

`tests/test_neural_network.py`에서는 보통 다음 동작을 확인한다.

- `params`에 필요한 파라미터 key가 존재하는가
- forward 출력 shape가 `(batch_size, 10)`인가
- softmax 출력의 행별 합이 1인가
- backward 후 `grads`에 파라미터별 gradient가 채워지는가
- `BatchNorm`, `Dropout`의 train 옵션이 올바르게 전달되는가

`NeuralNetwork` 테스트를 통과한다는 것은 개별 구성 요소들이 하나의 학습 가능한 모델로 연결되었다는 뜻이다.
