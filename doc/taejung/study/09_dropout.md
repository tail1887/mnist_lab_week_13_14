# Dropout 학습 문서

## 1. 핵심 역할

`Dropout`은 학습 중 일부 뉴런 출력을 무작위로 0으로 만들어 과적합을 줄이는 layer다.

모델이 특정 뉴런이나 특정 특징 조합에 너무 의존하면 학습 데이터에는 잘 맞지만 테스트 데이터에서는 성능이 떨어질 수 있다. Dropout은 매 학습 step마다 일부 뉴런을 꺼서 모델이 여러 특징을 분산해서 사용하도록 만든다.

MNIST 모델에서는 은닉층의 `ReLU` 뒤에 사용된다.

```text
Affine -> BatchNorm -> ReLU -> Dropout
```

## 배경: 왜 Dropout이 나왔는가

신경망은 파라미터 수가 많기 때문에 학습 데이터에 지나치게 맞춰질 수 있다. 이를 과적합이라고 한다.

과적합이 생기면 학습 데이터에서는 정확도가 높지만, 처음 보는 테스트 데이터에서는 성능이 떨어진다.

Dropout은 이 문제를 줄이기 위해 학습 중 일부 뉴런을 무작위로 꺼버린다.

```text
매 학습 step마다 다른 일부 뉴런 제거
-> 매번 조금 다른 신경망을 학습하는 효과
-> 특정 뉴런 조합에 대한 의존 감소
```

이 관점에서 Dropout은 여러 작은 모델을 동시에 학습하고 평균내는 ensemble과 비슷한 효과를 낸다고 볼 수 있다. 실제로 모든 가능한 mask 조합을 각각 학습시키는 것은 불가능하므로, 학습 중 랜덤 mask로 그 효과를 근사한다.

또 다른 직관은 `co-adaptation` 방지다. 어떤 뉴런들이 항상 함께 동작하면 서로에게 과하게 의존할 수 있다. Dropout은 매번 일부 뉴런을 끄기 때문에 각 뉴런이 다른 뉴런이 없더라도 쓸모 있는 특징을 만들도록 압박한다.

## 확률적 mask 공식

Dropout mask를 확률변수 `m`이라고 하자.

```text
m_i = 1  with probability 1 - drop_ratio
m_i = 0  with probability drop_ratio
```

학습 중 출력은 다음과 같다.

```text
y_i = x_i * m_i
```

현재 구현에서는 이 식을 그대로 사용한다.

```python
self.mask = np.random.rand(*x.shape) > self.drop_ratio
return x * self.mask
```

mask의 기대값은 다음과 같다.

```text
E[m_i] = 1 * (1 - drop_ratio) + 0 * drop_ratio
       = 1 - drop_ratio
```

따라서 학습 중 출력의 기대값은:

```text
E[y_i] = E[x_i * m_i]
       = x_i * E[m_i]
       = x_i * (1 - drop_ratio)
```

추론 때는 mask를 쓰지 않고 모든 뉴런을 사용한다. 그러면 출력이 학습 때의 평균보다 커질 수 있으므로, 현재 구현에서는 추론 시 `(1 - drop_ratio)`를 곱해 기대값을 맞춘다.

```text
train expected output = x * (1 - drop_ratio)
test output           = x * (1 - drop_ratio)
```

이것이 현재 코드의 추론 모드 scale이 나온 이유다.

## 2. Forward 흐름

현재 구현 파일은 `src/layers.py`다.

학습 모드에서는 랜덤 mask를 만든다.

```python
self.mask = np.random.rand(*x.shape) > self.drop_ratio
return x * self.mask
```

흐름은 다음과 같다.

1. 입력 `x`와 같은 shape의 랜덤 값을 만든다.
2. `drop_ratio`보다 큰 위치만 `True`로 둔다.
3. `False` 위치는 뉴런을 끈다.
4. `x * mask`로 일부 값을 0으로 만든다.

예를 들어 `drop_ratio=0.5`라면 대략 절반의 뉴런이 꺼진다.

```text
x    = [4, 5, 6, 7]
mask = [True, False, True, False]
out  = [4, 0, 6, 0]
```

## 3. 추론 모드 흐름

추론 모드에서는 랜덤하게 뉴런을 끄지 않는다.

```python
return x * (1 - self.drop_ratio)
```

현재 구현은 학습 중에는 살아남은 뉴런 값을 그대로 두고, 추론 시 전체 출력에 `(1 - drop_ratio)`를 곱하는 기본 dropout 방식이다.

학습 중에는 일부 뉴런만 살아남기 때문에 평균적인 출력 크기가 줄어든다. 추론에서는 모든 뉴런을 사용하므로, 학습 때의 평균 출력 크기와 맞추기 위해 scale을 적용한다.

```text
학습: 일부 뉴런만 사용
추론: 모든 뉴런 사용 + 출력 크기 조정
```

## 4. Backward 흐름

현재 구현은 다음과 같다.

```python
return dout * self.mask
```

Forward에서 꺼진 뉴런은 backward에서도 gradient가 흐르면 안 된다.

```text
mask가 False였던 위치 -> gradient 0
mask가 True였던 위치  -> gradient 통과
```

즉, Dropout은 forward와 backward에서 같은 mask를 사용한다.

예를 들어 forward 때 다음과 같았다면:

```text
mask = [True, False, True, False]
```

다음 층에서 온 gradient가 `[10, 20, 30, 40]`이어도 backward 결과는 다음과 같다.

```text
dx = [10, 0, 30, 0]
```

## Backward 공식 유도

학습 모드에서 Dropout은 다음 함수로 볼 수 있다.

```text
y = x * m
```

여기서 `m`은 forward 때 이미 정해진 mask다. backward를 계산할 때는 이 mask를 상수처럼 본다.

체인 룰을 적용하면:

```text
dL/dx = dL/dy * dy/dx
```

`y = x * m`이므로:

```text
dy/dx = m
```

따라서:

```text
dL/dx = dout * m
```

이 식이 코드의 다음 줄이다.

```python
return dout * self.mask
```

mask가 0인 위치는 forward에서 출력이 0이었고, backward에서도 gradient가 0이 된다. mask가 1인 위치는 gradient가 그대로 지나간다.

## 5. 구현 포인트

Dropout에서 가장 중요한 것은 학습 모드와 추론 모드가 다르다는 점이다.

```python
if train:
    self.mask = np.random.rand(*x.shape) > self.drop_ratio
    return x * self.mask
return x * (1 - self.drop_ratio)
```

`train=True`일 때만 랜덤 mask를 만든다. `predict()`처럼 평가하는 과정에서는 `train=False`로 실행되어야 한다.

`NeuralNetwork.forward()`는 `Dropout` layer를 만났을 때 `train` 값을 넘겨준다.

```python
if isinstance(layer, (BatchNorm, Dropout)):
    x = layer.forward(x, train=train)
```

## 6. BatchNorm과의 관계

BatchNorm과 Dropout은 목적이 다르다.

- `BatchNorm`: 값의 분포를 안정화해 학습을 쉽게 만든다.
- `Dropout`: 일부 뉴런을 꺼서 과적합을 줄인다.

둘을 함께 사용할 수 있지만, Dropout이 값을 랜덤하게 0으로 만들면 BatchNorm이 보는 분포가 흔들릴 수 있다. 그래서 현재 모델처럼 보통은 `Affine -> BatchNorm -> ReLU -> Dropout` 순서로 사용한다.

BatchNorm으로 먼저 분포를 안정화하고, ReLU로 비선형성을 넣은 뒤, Dropout으로 과적합을 완화하는 흐름이다.

## 7. 자주 헷갈리는 점

Dropout은 학습을 어렵게 만드는 것처럼 보일 수 있다. 실제로 학습 중에는 일부 정보를 일부러 제거한다.

하지만 이 제약 덕분에 모델이 특정 뉴런 하나에 의존하지 않고 여러 경로로 문제를 풀도록 학습된다. 이는 테스트 데이터에 대한 일반화 성능을 높이는 데 도움이 된다.

또한 현재 구현은 추론 시 scale을 적용하는 방식이다. 다른 구현에서는 학습 시 살아남은 뉴런을 `1 / (1 - drop_ratio)`로 키우고, 추론 때는 아무것도 하지 않는 inverted dropout 방식을 쓰기도 한다.

## 8. 테스트 관점

`tests/test_dropout.py`에서는 보통 다음 동작을 확인한다.

- 학습 모드에서 입력과 같은 shape의 mask가 만들어지는가
- 학습 모드에서 일부 값이 0이 되는가
- backward에서 forward 때 꺼진 위치의 gradient가 0이 되는가
- 추론 모드에서 랜덤 mask 없이 `(1 - drop_ratio)` scale이 적용되는가
- 출력 shape가 입력 shape와 같은가

`Dropout` 테스트를 통과한다는 것은 학습과 추론에서 서로 다른 동작이 올바르게 분리되었다는 뜻이다.
