# BatchNorm 학습 문서

## 1. 핵심 역할

`BatchNorm(Batch Normalization)`은 mini-batch 단위로 feature 값을 정규화해 학습을 안정화하는 layer다.

`Affine`을 통과한 값은 학습이 진행되면서 분포가 계속 바뀔 수 있다. 값의 스케일이 너무 커지거나 작아지면 gradient 흐름이 불안정해질 수 있다.

`BatchNorm`은 각 feature를 평균 0, 분산 1에 가깝게 맞춘다.

```text
x_norm = (x - mean) / sqrt(var + eps)
```

그 다음 학습 가능한 파라미터 `gamma`, `beta`를 적용한다.

```text
out = gamma * x_norm + beta
```

즉, 정규화로 분포를 안정화하면서도 모델이 필요한 scale과 shift를 다시 학습할 수 있게 한다.

## 배경: 왜 BatchNorm이 나왔는가

신경망의 각 층은 앞 층의 출력을 입력으로 받는다. 그런데 앞 층의 파라미터가 학습 중 계속 바뀌면, 뒤 층이 받는 입력 분포도 계속 바뀐다.

```text
앞 층 파라미터 변경
-> 앞 층 출력 분포 변경
-> 뒤 층 입력 분포 변경
-> 뒤 층이 다시 적응해야 함
```

이 문제를 흔히 `internal covariate shift`라고 설명한다. 실제로 BatchNorm의 효과를 이 용어 하나로만 설명하기는 어렵지만, 핵심 직관은 “각 층이 받는 입력 분포를 안정적으로 만들어 학습을 쉽게 한다”는 것이다.

입력 분포가 안정되면 learning rate를 더 크게 써도 학습이 덜 불안정해지고, gradient 흐름도 좋아질 수 있다.

## 정규화 공식 유도

한 mini-batch에서 특정 feature 하나만 보자. 값들이 다음과 같이 있다고 하자.

```text
x_1, x_2, ..., x_N
```

먼저 평균을 구한다.

```text
mu = (1/N) * sum_i x_i
```

평균을 빼면 값들의 중심이 0이 된다.

```text
x_centered_i = x_i - mu
```

그 다음 분산을 구한다.

```text
var = (1/N) * sum_i (x_i - mu)^2
```

표준편차로 나누면 분산이 1에 가까워진다.

```text
x_norm_i = (x_i - mu) / sqrt(var + eps)
```

`eps`는 분산이 0일 때 나누기 문제가 생기지 않도록 넣는 작은 값이다.

이렇게 하면 feature별로 batch 안에서 대략 다음 상태가 된다.

```text
mean(x_norm) ≈ 0
var(x_norm) ≈ 1
```

하지만 항상 평균 0, 분산 1만 강제하면 표현력이 제한될 수 있다. 그래서 학습 가능한 `gamma`, `beta`를 둔다.

```text
out_i = gamma * x_norm_i + beta
```

모델이 정규화된 값이 좋다고 판단하면 `gamma=1`, `beta=0` 근처를 유지할 수 있고, 다른 스케일이나 위치가 필요하면 직접 학습해서 바꿀 수 있다.

## 2. Forward 흐름

현재 구현 파일은 `src/layers.py`다.

학습 모드에서는 현재 mini-batch의 통계를 사용한다.

```python
mean = np.mean(x, axis=0)
var = np.var(x, axis=0)
self.x_centered = x - mean
self.std = np.sqrt(var + self.eps)
self.x_norm = self.x_centered / self.std
```

흐름은 다음과 같다.

1. feature별 평균 `mean`을 구한다.
2. feature별 분산 `var`를 구한다.
3. 입력에서 평균을 빼 `x_centered`를 만든다.
4. 표준편차 `std`로 나눠 `x_norm`을 만든다.
5. `gamma`, `beta`를 적용한다.

```python
return self.gamma * self.x_norm + self.beta
```

shape 관점에서는 다음과 같다.

```text
x:      (batch_size, feature_dim)
mean:   (feature_dim,)
var:    (feature_dim,)
gamma:  (feature_dim,)
beta:   (feature_dim,)
out:    (batch_size, feature_dim)
```

## 3. Running Statistics

학습 모드에서는 현재 batch 통계를 사용하면서 running 통계도 갱신한다.

```python
self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
```

추론 모드에서는 현재 batch의 평균과 분산을 사용하지 않는다.

```python
self.x_centered = x - self.running_mean
self.std = np.sqrt(self.running_var + self.eps)
self.x_norm = self.x_centered / self.std
```

추론할 때는 샘플이 하나씩 들어올 수도 있고, batch 구성이 매번 달라질 수 있다. 그래서 학습 중 누적한 `running_mean`, `running_var`를 사용한다.

## 4. Backward 흐름

현재 구현은 다음과 같다.

```python
batch_size = dout.shape[0]
self.dbeta = np.sum(dout, axis=0)
self.dgamma = np.sum(dout * self.x_norm, axis=0)

dx_norm = dout * self.gamma
dx = (
    dx_norm
    - np.mean(dx_norm, axis=0)
    - self.x_norm * np.mean(dx_norm * self.x_norm, axis=0)
) / self.std
return dx
```

`beta`는 출력에 그대로 더해지는 값이므로 gradient는 batch 방향 합이다.

```text
dbeta = sum(dout)
```

`gamma`는 `x_norm`에 곱해지는 값이므로 `dout * x_norm`을 batch 방향으로 합한다.

```text
dgamma = sum(dout * x_norm)
```

`dx`는 정규화 과정의 미분을 한 줄로 정리한 형태다.

```text
dx = (dx_norm - mean(dx_norm) - x_norm * mean(dx_norm * x_norm)) / std
```

이 식은 평균을 빼고 분산으로 나눈 정규화 과정 전체를 반영한다.

## Backward 공식 직관

BatchNorm backward가 복잡한 이유는 한 샘플의 출력이 자기 자신만이 아니라 batch 전체의 평균과 분산에 의존하기 때문이다.

```text
x_norm_i = (x_i - mu) / std
mu = mean(x)
std = sqrt(var + eps)
```

`x_i`가 바뀌면 세 가지 경로로 loss에 영향을 준다.

```text
1. x_i가 직접 x_norm_i에 주는 영향
2. x_i가 batch 평균 mu를 바꿔 모든 샘플에 주는 영향
3. x_i가 batch 분산 var를 바꿔 모든 샘플에 주는 영향
```

그래서 단순히 `dout / std`만 하면 부족하다. 평균을 통해 생기는 영향과 분산을 통해 생기는 영향을 빼줘야 한다.

현재 구현의 축약식은 이 세 경로를 정리한 결과다.

```text
dx = (dx_norm - mean(dx_norm) - x_norm * mean(dx_norm * x_norm)) / std
```

각 항의 의미는 다음과 같다.

```text
dx_norm:
  정규화된 값으로 직접 들어온 gradient

- mean(dx_norm):
  평균 mu를 통해 batch 전체에 퍼지는 영향을 보정

- x_norm * mean(dx_norm * x_norm):
  분산 var를 통해 batch 전체에 퍼지는 영향을 보정

/ std:
  표준편차로 나눈 정규화 스케일 반영
```

즉, BatchNorm backward는 “현재 샘플의 직접 영향”에서 “batch 통계가 변해서 생기는 간접 영향”을 함께 반영한 미분이다.

## 5. 구현 포인트

Forward에서 저장한 값들이 backward에 필요하다.

```python
self.x_centered
self.std
self.x_norm
```

특히 `x_norm`은 `dgamma` 계산에 필요하고, `std`는 `dx` 계산에 필요하다.

또한 `eps`는 분산이 0에 가까울 때 나누기 문제가 생기지 않도록 한다.

```python
self.eps = 1e-7
```

`eps`가 없으면 어떤 feature가 batch 안에서 모두 같은 값을 가질 때 `sqrt(var)`가 0이 되어 계산이 깨질 수 있다.

## 6. 자주 헷갈리는 점

BatchNorm은 단순히 평균 0, 분산 1로 만드는 layer가 아니다. 그 뒤에 `gamma`, `beta`를 학습해서 모델이 필요한 분포를 다시 선택할 수 있다.

```text
정규화 -> 안정화
gamma, beta -> 표현력 보존
```

또한 학습 모드와 추론 모드가 다르다.

- 학습: 현재 batch 통계 사용, running 통계 갱신
- 추론: running 통계 사용

이 구분이 없으면 평가할 때 결과가 batch 구성에 따라 흔들릴 수 있다.

## 7. 테스트 관점

`tests/test_batchnorm.py`에서는 보통 다음 동작을 확인한다.

- 학습 모드 forward에서 출력이 정규화되는가
- `gamma`, `beta`가 출력에 반영되는가
- `running_mean`, `running_var`가 갱신되는가
- 추론 모드에서 running 통계를 사용하는가
- backward에서 `dx`, `dgamma`, `dbeta` shape가 올바른가

`BatchNorm` 테스트를 통과한다는 것은 학습 안정화 layer가 forward와 backward 양쪽에서 일관되게 동작한다는 뜻이다.
