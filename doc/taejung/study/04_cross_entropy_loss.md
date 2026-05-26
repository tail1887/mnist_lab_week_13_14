# Cross Entropy Loss 학습 문서

## 1. 핵심 역할

`cross_entropy_loss`는 모델의 예측 확률이 정답과 얼마나 다른지 계산하는 손실 함수다.

MNIST는 0부터 9까지 10개 클래스 중 하나를 맞히는 다중 클래스 분류 문제다. 이때 모델은 `Softmax`를 통해 각 클래스의 확률을 출력하고, `Cross Entropy Loss`는 정답 클래스의 확률이 얼마나 높은지를 기준으로 손실을 계산한다.

정답 클래스 확률이 높으면 loss가 작고, 정답 클래스 확률이 낮으면 loss가 커진다.

```text
loss = -log(정답 클래스 확률)
```

## 배경: 왜 Cross Entropy를 쓰는가

분류 문제에서 모델의 출력은 “각 클래스일 확률”로 해석한다. 이때 정답 클래스에 높은 확률을 주면 좋은 예측이고, 낮은 확률을 주면 나쁜 예측이다.

단순히 정답 여부만 0 또는 1로 보는 accuracy는 학습에 직접 쓰기 어렵다. 예를 들어 정답 클래스 확률이 `0.51`에서 `0.99`로 좋아져도 둘 다 정답이면 accuracy는 똑같다. 반대로 `0.49`와 `0.01`은 둘 다 오답일 수 있지만, 모델의 나쁜 정도는 다르다.

Cross Entropy는 정답 클래스 확률 자체를 손실로 사용한다.

```text
p = 정답 클래스에 모델이 준 확률
loss = -log(p)
```

이 함수는 다음 성질을 가진다.

```text
p가 1에 가까움 -> -log(p)는 0에 가까움
p가 0에 가까움 -> -log(p)는 매우 커짐
```

즉, 정답에 확신을 줄수록 손실이 작아지고, 정답에 낮은 확률을 줄수록 강하게 벌점을 받는다.

## 정보이론과 최대우도 관점

Cross Entropy는 정보이론에서 두 확률분포의 차이를 재는 식에서 나온다.

정답 분포를 `t`, 모델 예측 분포를 `y`라고 하면 cross entropy는 다음과 같다.

```text
H(t, y) = -sum_i t_i log(y_i)
```

분류 문제에서 정답은 보통 one-hot 분포다. 예를 들어 정답이 숫자 3이면:

```text
t = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
```

이 경우 `t_i`가 1인 항만 남는다.

```text
H(t, y)
= -sum_i t_i log(y_i)
= -log(y_정답)
```

그래서 구현에서는 모든 클래스의 log를 다 더하지 않고 정답 클래스 확률만 뽑아 `-log`를 계산한다.

최대우도 관점에서도 같은 식이 나온다. 모델이 정답 라벨을 맞힐 확률을 최대화하고 싶다면 다음 값을 최대화해야 한다.

```text
P(정답 | 입력) = y_정답
```

여러 샘플에 대해서는 확률을 곱한다.

```text
Likelihood = product_n y_{n, 정답}
```

곱은 다루기 어려우므로 log를 취한다.

```text
log Likelihood = sum_n log(y_{n, 정답})
```

학습에서는 보통 손실을 최소화하므로 음수를 붙인다.

```text
Negative Log Likelihood = -sum_n log(y_{n, 정답})
```

이것이 Cross Entropy Loss로 이어진다.

## 2. Forward 흐름

현재 구현 파일은 `src/losses.py`다.

```python
batch_size = y_pred.shape[0]
return -np.sum(np.log(y_pred[np.arange(batch_size), y_true] + 1e-7)) / batch_size
```

입력은 다음과 같다.

```text
y_pred: (batch_size, 10)  # Softmax 출력 확률
y_true: (batch_size,)     # 정답 라벨
```

흐름은 다음과 같다.

1. `np.arange(batch_size)`로 각 샘플의 행 index를 만든다.
2. `y_true`로 정답 클래스 열 index를 고른다.
3. `y_pred[np.arange(batch_size), y_true]`로 정답 클래스 확률만 뽑는다.
4. 정답 확률에 `log`를 취한다.
5. 음수를 붙여 손실로 만든다.
6. batch 평균을 반환한다.

예를 들어 batch 크기가 3이고 정답이 `[2, 0, 1]`이면 다음 위치를 고른다.

```text
y_pred[0, 2]
y_pred[1, 0]
y_pred[2, 1]
```

## 3. Backward 흐름

`cross_entropy_loss` 함수 안에는 backward 함수가 따로 없다.

이 과제에서는 `Softmax + Cross Entropy`의 결합 gradient를 `src/training.py`에서 직접 만든다.

```python
dout = y_pred.copy()
dout[np.arange(current_batch_size), y_batch] -= 1
dout /= current_batch_size
```

이는 다음 수식과 같다.

```text
dout = (y_pred - one_hot(y_true)) / batch_size
```

`Softmax`와 `Cross Entropy`를 따로 미분하면 식이 복잡하지만, 두 함수를 합쳐 미분하면 위처럼 간단해진다.

그래서 훈련 루프에서는 loss 값을 계산한 뒤, 이 gradient를 만들어 `model.backward(dout)`으로 넘긴다.

## Softmax와 결합한 미분 유도

Softmax 출력은 다음과 같다.

```text
y_i = exp(z_i) / sum_k exp(z_k)
```

Cross Entropy는 one-hot 정답 `t`에 대해 다음과 같다.

```text
L = -sum_i t_i log(y_i)
```

Softmax의 log를 전개하면:

```text
log(y_i)
= log(exp(z_i) / sum_k exp(z_k))
= z_i - log(sum_k exp(z_k))
```

이를 loss에 대입한다.

```text
L = -sum_i t_i (z_i - log(sum_k exp(z_k)))
```

one-hot 정답에서는 `sum_i t_i = 1`이므로:

```text
L = -sum_i t_i z_i + log(sum_k exp(z_k))
```

이제 logit `z_j`에 대해 미분한다.

첫 번째 항:

```text
d/dz_j [-sum_i t_i z_i] = -t_j
```

두 번째 항:

```text
d/dz_j [log(sum_k exp(z_k))]
= exp(z_j) / sum_k exp(z_k)
= y_j
```

따라서:

```text
dL/dz_j = y_j - t_j
```

batch 평균을 쓰면 batch size로 나눈다.

```text
dL/dz = (y - t) / batch_size
```

현재 코드에서 정답 위치에 1을 빼고 batch 크기로 나누는 이유가 이 미분 결과다.

## 4. 구현 포인트

가장 중요한 구현 포인트는 정답 클래스의 확률만 선택하는 부분이다.

```python
y_pred[np.arange(batch_size), y_true]
```

`y_true`가 one-hot 벡터가 아니라 정수 라벨이라는 점도 중요하다.

```text
정수 라벨: [3, 1, 7]
one-hot:  [[0,0,0,1,0,0,0,0,0,0], ...]
```

현재 구현은 정수 라벨을 직접 index로 사용한다.

또한 `log(0)`을 피하기 위해 아주 작은 값인 `1e-7`을 더한다.

```python
np.log(... + 1e-7)
```

확률이 0에 가까우면 loss가 매우 커져야 하지만, 정확히 0이면 `log(0)` 때문에 계산이 깨질 수 있다.

## 5. 자주 헷갈리는 점

Cross Entropy Loss는 모든 클래스 확률을 직접 비교하는 것처럼 보일 수 있지만, 정수 라벨을 사용하는 구현에서는 정답 클래스 확률만 직접 뽑아 loss를 계산한다.

오답 클래스 확률은 softmax의 합이 1이라는 제약을 통해 간접적으로 영향을 준다. 정답 클래스 확률이 높아지려면 다른 클래스 확률은 상대적으로 낮아져야 하기 때문이다.

또한 loss는 batch 전체의 합이 아니라 평균이다.

```python
/ batch_size
```

평균을 사용하면 batch 크기가 달라져도 loss 크기를 비교하기 쉽다.

## 6. 테스트 관점

`tests/test_cross_entropy_loss.py`에서는 보통 다음 동작을 확인한다.

- 정답 클래스 확률을 올바르게 선택하는가
- `-log(probability)` 계산이 맞는가
- batch 평균을 반환하는가
- 확률이 0에 가까워도 계산이 깨지지 않는가
- 1개 샘플 입력도 처리할 수 있는가

`cross_entropy_loss`는 학습 로그에서 보는 loss 값의 기준이 되므로, 이 함수가 틀리면 학습이 좋아지는지 나빠지는지 판단하기 어렵다.
