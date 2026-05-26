# Adam 학습 문서

## 1. 핵심 역할

`Adam`은 gradient의 이동평균과 gradient 제곱의 이동평균을 함께 사용해 파라미터를 업데이트하는 optimizer다.

SGD는 모든 파라미터에 같은 learning rate를 적용한다.

```text
parameter = parameter - lr * gradient
```

Adam은 파라미터마다 gradient의 흐름을 기록하고, 업데이트 크기를 조절한다. 그래서 SGD보다 더 빠르고 안정적으로 loss가 줄어드는 경우가 많다.

## 배경: SGD의 한계와 Adam의 등장

SGD는 단순하고 강력하지만 몇 가지 한계가 있다.

첫째, 모든 파라미터에 같은 learning rate를 적용한다.

```text
W_new = W - lr * gradient
```

어떤 파라미터는 gradient가 자주 크고, 어떤 파라미터는 gradient가 작거나 드물게 나타날 수 있다. 그런데 같은 learning rate를 적용하면 파라미터별 상황을 반영하지 못한다.

둘째, gradient 방향이 지그재그로 흔들릴 수 있다. 특히 loss 지형이 좁고 긴 골짜기처럼 생긴 경우, SGD는 양옆으로 크게 흔들리면서 천천히 내려갈 수 있다.

이 문제를 완화하기 위해 두 흐름이 나왔다.

```text
Momentum: gradient의 방향 평균을 사용해 관성을 만든다.
RMSProp: gradient 제곱 평균을 사용해 파라미터별 이동 크기를 조절한다.
```

Adam은 이 둘을 합친 방식으로 볼 수 있다.

```text
Adam = Momentum + RMSProp + Bias Correction
```

이름도 `Adaptive Moment Estimation`에서 왔다. gradient의 1차 모멘트와 2차 모멘트를 추정해서 적응적으로 업데이트한다는 뜻이다.

## Momentum에서 온 1차 모멘트

Momentum은 지금 gradient만 보는 대신, 이전 gradient들의 이동평균을 사용한다.

```text
m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
```

여기서 `g_t`는 현재 gradient다.

gradient가 계속 같은 방향이면 `m_t`가 커져 그 방향으로 더 꾸준히 이동한다. 반대로 방향이 자주 바뀌는 성분은 평균 과정에서 상쇄된다.

현재 코드에서는 `beta1 = 0.9`에 해당하는 값을 사용한다.

```python
self.m[key] = self.m.get(key, 0) * 0.9 + grads[key] * 0.1
```

## RMSProp에서 온 2차 모멘트

RMSProp은 gradient 제곱의 이동평균을 저장한다.

```text
v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
```

gradient가 큰 파라미터는 `v_t`가 커지고, 업데이트할 때 `sqrt(v_t)`로 나누므로 이동 폭이 줄어든다.

```text
update 크기 ≈ g_t / sqrt(v_t)
```

즉, 자주 크게 변하는 파라미터는 조심스럽게 움직이고, gradient가 작은 파라미터는 상대적으로 더 움직일 수 있게 된다.

현재 코드에서는 `beta2`도 `0.9`로 단순화되어 있다.

```python
self.v[key] = self.v.get(key, 0) * 0.9 + grads[key]**2 * 0.1
```

## 2. Update 흐름

현재 구현 파일은 `src/optimizers.py`다.

```python
self.t += 1
for key in params.keys():
    self.m[key] = self.m.get(key, 0) * 0.9 + grads[key] * 0.1
    self.v[key] = self.v.get(key, 0) * 0.9 + grads[key]**2 * 0.1
    m_hat = self.m[key] / (1 - 0.9**self.t)
    v_hat = self.v[key] / (1 - 0.9**self.t)
    params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + 1e-8)
```

흐름은 다음과 같다.

1. 업데이트 횟수 `t`를 1 증가시킨다.
2. 각 파라미터의 gradient 이동평균 `m`을 갱신한다.
3. 각 파라미터의 gradient 제곱 이동평균 `v`를 갱신한다.
4. 초반에 `m`, `v`가 0에 치우치는 문제를 bias correction으로 보정한다.
5. `m_hat / sqrt(v_hat)` 형태로 파라미터 업데이트 크기를 조절한다.

## 3. 1차 모멘트와 2차 모멘트

현재 코드에서 `m`은 gradient의 이동평균이다.

```python
self.m[key] = self.m.get(key, 0) * 0.9 + grads[key] * 0.1
```

이는 최근 gradient 방향을 부드럽게 누적한다. gradient가 계속 비슷한 방향을 가리키면 `m`도 그 방향으로 커진다.

`v`는 gradient 제곱의 이동평균이다.

```python
self.v[key] = self.v.get(key, 0) * 0.9 + grads[key]**2 * 0.1
```

이는 gradient 크기를 누적한다. 어떤 파라미터의 gradient가 계속 크면 `v`도 커진다.

업데이트할 때는 `sqrt(v_hat)`으로 나누기 때문에, gradient가 큰 파라미터는 이동 폭이 조절된다.

## 4. Bias Correction

Adam의 `m`과 `v`는 처음에 비어 있고, 코드에서는 기본값 0에서 시작한다.

```python
self.m, self.v = {}, {}
```

초반에는 이동평균 값이 실제보다 작게 추정될 수 있다. 그래서 현재 코드에서는 다음처럼 보정한다.

```python
m_hat = self.m[key] / (1 - 0.9**self.t)
v_hat = self.v[key] / (1 - 0.9**self.t)
```

`t`가 작을 때는 보정 효과가 크고, 학습이 진행되어 `t`가 커질수록 보정 효과가 줄어든다.

일반적인 Adam 공식에서는 `m`과 `v`에 서로 다른 계수인 `beta1`, `beta2`를 쓰는 경우가 많다. 현재 과제 구현은 둘 다 `0.9`를 사용해 단순화되어 있다.

Bias correction이 필요한 이유를 간단히 보면, 처음에는 `m_0 = 0`이다.

첫 번째 step에서:

```text
m_1 = beta * 0 + (1 - beta) * g_1
```

`beta = 0.9`이면:

```text
m_1 = 0.1 * g_1
```

실제 gradient보다 10분의 1로 작게 시작한다. 그래서 `1 - beta^t`로 나누어 초반의 0 쪽 치우침을 보정한다.

```text
m_hat = m_t / (1 - beta^t)
```

첫 번째 step에서는:

```text
m_hat = (0.1 * g_1) / (1 - 0.9^1)
      = (0.1 * g_1) / 0.1
      = g_1
```

이렇게 초반에도 이동평균이 지나치게 작게 잡히지 않도록 만든다.

## 5. SGD와의 차이

SGD는 현재 gradient만 보고 바로 이동한다.

```text
SGD: 지금 gradient만 사용
```

Adam은 과거 gradient의 흐름까지 함께 본다.

```text
Adam: 지금 gradient + 과거 gradient 평균 + gradient 크기 평균
```

그래서 gradient 방향이 많이 흔들릴 때도 업데이트가 비교적 안정적이다. 보고서 실험에서도 Adam은 SGD보다 빠르게 높은 정확도에 도달했다.

## 6. 구현 포인트

`m`과 `v`는 파라미터별로 따로 저장해야 한다.

```python
self.m[key]
self.v[key]
```

`W1`, `b1`, `W2` 같은 파라미터는 shape도 다르고 gradient 크기도 다르다. 따라서 하나의 평균값을 공유하면 안 되고, key별로 따로 관리해야 한다.

또한 분모에 작은 값 `1e-8`을 더한다.

```python
np.sqrt(v_hat) + 1e-8
```

이는 `v_hat`이 0에 가까울 때 0으로 나누는 문제를 막기 위한 수치 안정성 장치다.

## 7. 자주 헷갈리는 점

Adam의 learning rate도 여전히 중요하다. Adam이 파라미터별 이동 크기를 조절해주지만, 전체적인 이동 규모는 `self.lr`이 정한다.

또한 Adam은 상태를 가진 optimizer다. `self.m`, `self.v`, `self.t`가 계속 누적되므로, 학습 도중 optimizer 객체를 새로 만들면 이전까지의 이동평균 정보가 사라진다.

## 8. 테스트 관점

`tests/test_adam.py`에서는 보통 다음 동작을 확인한다.

- 첫 번째 update 후 `m`, `v`, `t`가 생성되는가
- bias correction이 적용되는가
- 파라미터가 gradient 반대 방향으로 업데이트되는가
- 여러 번 update할 때 `m`, `v`가 누적되는가
- 여러 파라미터 key를 각각 따로 관리하는가

Adam 테스트를 통과한다는 것은 단순히 파라미터가 바뀌는 것뿐 아니라, optimizer 내부 상태가 올바르게 누적된다는 뜻이다.
