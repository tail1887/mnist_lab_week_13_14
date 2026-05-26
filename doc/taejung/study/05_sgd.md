# SGD 학습 문서

## 1. 핵심 역할

`SGD(Stochastic Gradient Descent)`는 gradient를 이용해 파라미터를 업데이트하는 가장 기본적인 optimizer다.

손실을 줄이려면 파라미터를 gradient의 반대 방향으로 이동시킨다.

```text
parameter = parameter - learning_rate * gradient
```

gradient는 loss가 가장 빠르게 증가하는 방향을 가리킨다. 따라서 그 반대 방향으로 이동하면 loss가 줄어드는 방향으로 파라미터를 바꾸는 것이다.

## 배경: 왜 gradient 반대 방향으로 가는가

신경망 학습의 목표는 loss 함수 `L(W)`를 최소화하는 파라미터 `W`를 찾는 것이다.

loss가 어떤 파라미터 `w`에 대한 함수라고 하자. 현재 위치가 `w`일 때, 아주 작은 변화 `delta`를 주면 테일러 전개로 다음처럼 근사할 수 있다.

```text
L(w + delta) ≈ L(w) + dL/dw * delta
```

여기서 `dL/dw`가 양수라면 `delta`를 음수로 잡아야 loss가 줄어든다.

```text
dL/dw > 0 이면 w를 줄임
dL/dw < 0 이면 w를 키움
```

이를 하나의 식으로 쓰면:

```text
delta = -lr * dL/dw
```

따라서 업데이트 식은 다음이 된다.

```text
w_new = w + delta
      = w - lr * dL/dw
```

파라미터가 벡터나 행렬이어도 같은 원리가 적용된다.

```text
W_new = W - lr * gradient
```

gradient는 loss가 가장 빠르게 증가하는 방향이다. 음수를 붙이면 가장 빠르게 감소하는 방향이 된다.

## Stochastic의 의미

전체 데이터셋의 loss를 정확히 계산하려면 모든 학습 데이터를 사용해야 한다.

```text
L = (1/N) * sum_n L_n
```

하지만 데이터가 많으면 매 업데이트마다 전체 데이터를 쓰는 것이 느리다. 그래서 일부 mini-batch만 뽑아 gradient를 근사한다.

```text
전체 gradient ≈ mini-batch gradient
```

이 근사는 매번 조금씩 흔들리지만, 계산이 빠르고 자주 업데이트할 수 있다. `Stochastic`이라는 이름은 이처럼 랜덤하게 뽑은 데이터 일부로 gradient를 추정한다는 뜻이다.

현재 코드에서는 optimizer가 랜덤 샘플링을 직접 하지 않고, `train()`에서 매 epoch마다 데이터를 섞고 mini-batch를 만든다.

## 2. Update 흐름

현재 구현 파일은 `src/optimizers.py`다.

```python
for key in grads.keys():
    params[key] -= self.lr * grads[key]
```

입력은 다음과 같다.

```text
params: 모델의 파라미터 dict
grads: 각 파라미터에 대한 gradient dict
```

예를 들어 `NeuralNetwork`의 `params`에는 다음 값들이 들어 있다.

```text
W1, b1, gamma1, beta1, W2, b2, gamma2, beta2, W3, b3
```

`grads`에는 같은 key에 대응하는 gradient가 들어 있다.

```text
dW1 -> grads["W1"]
db1 -> grads["b1"]
dgamma1 -> grads["gamma1"]
dbeta1 -> grads["beta1"]
```

SGD는 각 key를 돌면서 같은 이름의 gradient를 찾아 파라미터를 직접 갱신한다.

## 3. Learning Rate 의미

`learning_rate` 또는 `lr`은 한 번 업데이트할 때 얼마나 크게 이동할지 정하는 값이다.

```python
self.lr = lr
```

학습률이 너무 크면 최소점을 지나쳐 loss가 불안정해질 수 있다.

```text
이동 폭이 너무 큼 -> loss가 튐 -> 수렴 실패 가능
```

학습률이 너무 작으면 loss는 안정적으로 줄 수 있지만 학습이 매우 느리다.

```text
이동 폭이 너무 작음 -> 학습 속도 느림 -> epoch를 많이 써야 함
```

보고서 실험에서도 SGD에서 learning rate를 `0.001`에서 `0.01`로 올렸을 때 정확도가 크게 개선되었다. 이는 기존 설정에서는 이동 폭이 너무 작아 충분히 빠르게 수렴하지 못했다는 의미로 볼 수 있다.

## 4. Backward와의 연결

SGD 자체는 gradient를 계산하지 않는다. gradient는 각 layer의 backward에서 계산된다.

전체 흐름은 다음과 같다.

```text
model.forward()
-> loss 계산
-> model.backward()
-> model.grads 채움
-> optimizer.update(model.params, model.grads)
```

즉, SGD는 `model.grads`에 이미 들어 있는 값을 믿고 파라미터를 갱신하는 역할만 한다.

## 5. 구현 포인트

현재 구현은 in-place update를 사용한다.

```python
params[key] -= self.lr * grads[key]
```

이 코드는 `params[key]` 배열 자체를 수정한다. `Affine`이나 `BatchNorm` layer는 `self.params` 안의 배열 객체를 공유하고 있으므로, optimizer가 `params`를 바꾸면 layer가 참조하는 파라미터도 함께 바뀐다.

이 공유 구조 덕분에 업데이트 후 다음 forward에서 새 파라미터가 바로 사용된다.

## 6. 자주 헷갈리는 점

SGD는 이름에 `Stochastic`이 들어가지만, 현재 `SGD.update()` 함수 안에서 데이터를 랜덤하게 고르지는 않는다.

랜덤성은 `train()` 함수에서 mini-batch를 만들 때 들어간다.

```python
indices = np.random.permutation(len(x_train))
```

SGD optimizer는 그 mini-batch에서 계산된 gradient를 받아 업데이트만 수행한다.

또한 `params`와 `grads`의 key가 맞아야 한다. key가 빠지거나 이름이 다르면 어떤 파라미터는 업데이트되지 않거나 오류가 날 수 있다.

## 7. 테스트 관점

`tests/test_sgd.py`에서는 보통 다음 동작을 확인한다.

- `params[key]`가 `params[key] - lr * grads[key]`로 바뀌는가
- 여러 파라미터 key를 모두 업데이트하는가
- 원본 `params` dict가 in-place로 갱신되는가
- learning rate 값이 업데이트 크기에 반영되는가

SGD 테스트를 통과한다는 것은 backward에서 계산한 gradient가 실제 파라미터 변화로 이어질 준비가 되었다는 뜻이다.
