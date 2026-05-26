# Softmax 학습 문서

## 1. 핵심 역할

`Softmax`는 출력층의 점수인 `logit`을 클래스별 확률로 바꾼다.

MNIST 분류에서는 출력 차원이 10이다. 신경망의 마지막 `Affine` 층은 각 숫자 클래스에 대한 점수를 만든다.

```text
[2.1, 0.3, -1.2, ...]  ->  [0.65, 0.11, 0.02, ...]
```

`Softmax`를 통과한 값은 다음 성질을 가진다.

- 모든 값이 0 이상이다.
- 한 샘플의 클래스 확률 합이 1이다.
- 값이 클수록 해당 클래스로 예측할 가능성이 높다.

## 배경: 왜 Softmax가 필요한가

이진 분류에서는 `sigmoid` 하나로 어떤 클래스일 확률을 표현할 수 있다.

```text
sigmoid(z) = 1 / (1 + exp(-z))
```

하지만 MNIST는 10개 숫자 중 하나를 고르는 문제다. 출력이 10개일 때 각 점수를 독립적으로 sigmoid에 넣으면 각 값은 0과 1 사이가 되지만, 10개 값의 합이 1이라는 보장은 없다.

```text
sigmoid를 10개에 각각 적용한 경우:
[0.8, 0.7, 0.2, ...]  # 합이 1이 아닐 수 있음
```

다중 클래스 분류에서는 “10개 중 하나”를 골라야 하므로, 전체 클래스 확률의 합이 1이어야 한다. Softmax는 임의의 실수 점수들을 확률분포로 바꾸기 위해 사용된다.

또한 `exp`를 쓰는 이유는 점수 차이를 양수 비율로 바꾸기 위해서다.

```text
score가 큰 클래스 -> exp(score)가 훨씬 큼 -> 확률도 커짐
score가 작은 클래스 -> exp(score)가 작음 -> 확률도 작아짐
```

즉, Softmax는 점수의 상대적 크기를 확률의 상대적 크기로 바꾸는 함수다.

## 공식 유도

모델의 마지막 층이 만든 점수를 `z`라고 하자. 이 점수는 아직 확률이 아니다.

```text
z = [z_1, z_2, ..., z_K]
```

확률로 쓰려면 두 조건이 필요하다.

```text
1. 각 값은 0 이상이어야 한다.
2. 전체 합은 1이어야 한다.
```

`exp(z_i)`는 항상 양수이므로 첫 번째 조건을 만족한다.

```text
exp(z_i) > 0
```

그 다음 전체 합으로 나누면 두 번째 조건도 만족한다.

```text
y_i = exp(z_i) / sum_j exp(z_j)
```

합을 확인하면 다음과 같다.

```text
sum_i y_i
= sum_i exp(z_i) / sum_j exp(z_j)
= sum_i exp(z_i) / sum_i exp(z_i)
= 1
```

그래서 Softmax 출력은 확률분포로 해석할 수 있다.

## 2. Forward 흐름

현재 구현 파일은 `src/activations.py`다.

```python
x_shifted = x - np.max(x, axis=1, keepdims=True)
exp_x = np.exp(x_shifted)
sum_exp_x = np.sum(exp_x, axis=1, keepdims=True)
out = exp_x / sum_exp_x
```

흐름은 다음과 같다.

1. 각 샘플마다 가장 큰 logit 값을 구한다.
2. 모든 logit에서 그 최댓값을 뺀다.
3. `exp`를 적용한다.
4. 각 샘플별 `exp` 합으로 나누어 확률로 만든다.

수식으로 쓰면 다음과 같다.

```text
softmax(x_i) = exp(x_i) / sum(exp(x_j))
```

여기서 `i`는 특정 클래스, `j`는 모든 클래스를 의미한다.

## 3. 수치 안정성

현재 구현에서 가장 중요한 부분은 이 줄이다.

```python
x_shifted = x - np.max(x, axis=1, keepdims=True)
```

`exp`는 입력이 조금만 커져도 값이 매우 커진다. 예를 들어 `np.exp(1000)`은 overflow를 일으킬 수 있다.

하지만 softmax는 모든 값에서 같은 상수를 빼도 결과가 변하지 않는다.

```text
softmax(x) == softmax(x - C)
```

이 성질은 식으로도 확인할 수 있다.

```text
softmax(z_i - C)
= exp(z_i - C) / sum_j exp(z_j - C)
= exp(z_i) * exp(-C) / sum_j (exp(z_j) * exp(-C))
= exp(z_i) / sum_j exp(z_j)
= softmax(z_i)
```

그래서 각 행의 최댓값을 빼면 가장 큰 값이 0이 되고, 나머지는 0 이하가 된다. 이렇게 하면 `exp` 계산이 안정적이다.

`axis=1`을 사용하는 이유는 MNIST 입력이 보통 `(batch_size, num_classes)` shape이기 때문이다. 즉, 샘플마다 따로 최댓값을 구해야 한다.

```text
x shape = (batch_size, 10)
axis=1   = 각 샘플 안에서 10개 클래스 방향
```

## 4. Backward 흐름

현재 구현은 다음과 같다.

```python
return dout
```

이 코드만 보면 softmax 미분을 하지 않는 것처럼 보일 수 있다. 하지만 이 과제에서는 `Softmax`와 `Cross Entropy`를 합친 gradient를 `train()` 함수에서 직접 만든다.

`src/training.py`에서는 다음 방식으로 출력층 gradient를 만든다.

```python
dout = y_pred.copy()
dout[np.arange(current_batch_size), y_batch] -= 1
dout /= current_batch_size
```

이는 `Softmax + Cross Entropy`를 함께 미분했을 때 나오는 간단한 형태다.

```text
dout = (y_pred - one_hot(y_true)) / batch_size
```

따라서 `Softmax.backward()`는 이미 계산된 gradient를 그대로 이전 층으로 넘기면 된다.

## Softmax 미분과 Cross Entropy 연결

Softmax 자체의 미분은 단순한 원소별 미분이 아니다. 하나의 출력 `y_i`가 모든 입력 `z_j`의 영향을 받기 때문이다.

```text
y_i = exp(z_i) / sum_k exp(z_k)
```

미분하면 두 경우로 나뉜다.

```text
i == j: dy_i/dz_j = y_i * (1 - y_i)
i != j: dy_i/dz_j = -y_i * y_j
```

이를 하나의 식으로 쓰면 다음과 같다.

```text
dy_i/dz_j = y_i * (delta_ij - y_j)
```

여기서 `delta_ij`는 `i == j`이면 1, 아니면 0인 값이다.

Softmax만 따로 미분하면 이렇게 Jacobian 행렬이 필요해서 복잡하다. 하지만 Cross Entropy와 함께 쓰면 식이 크게 단순해진다.

정답 one-hot 벡터를 `t`라고 하고 loss를 다음처럼 두면:

```text
L = -sum_i t_i log(y_i)
```

Softmax까지 포함한 logit `z_j`에 대한 미분은 다음이 된다.

```text
dL/dz_j = y_j - t_j
```

현재 `train()` 함수에서 `dout = y_pred - one_hot(y_true)` 형태를 만드는 이유가 바로 이 결과 때문이다.

## 5. 구현 포인트

`np.max(x)`처럼 전체 배열에서 하나의 최댓값만 구하면 여러 샘플이 하나의 기준값을 공유하게 된다.

현재 구현처럼 row별로 계산해야 한다.

```python
np.max(x, axis=1, keepdims=True)
```

`keepdims=True`를 쓰면 결과 shape가 `(batch_size, 1)`로 유지된다. 덕분에 원래 입력 `(batch_size, 10)`과 broadcasting이 자연스럽게 맞는다.

## 6. 자주 헷갈리는 점

`Softmax`의 출력은 확률처럼 해석할 수 있지만, 모델이 실제로 확신한다는 뜻은 아니다. 단지 10개 클래스 점수를 합이 1인 형태로 정규화한 값이다.

또한 `Softmax.backward()`가 단순히 `dout`을 반환한다고 해서 역전파가 생략된 것은 아니다. 이 구현에서는 손실 함수와 결합된 미분식을 앞에서 만들어 넘기는 구조다.

## 7. 테스트 관점

`tests/test_softmax.py`에서는 보통 다음 동작을 확인한다.

- 각 행의 확률 합이 1인가
- 출력값이 모두 0 이상인가
- 큰 입력값에서도 overflow 없이 계산되는가
- batch 입력에서 샘플별로 독립적으로 softmax가 적용되는가
- backward가 입력 gradient를 그대로 반환하는가

`Softmax` 테스트에서 중요한 것은 단일 벡터뿐 아니라 여러 샘플이 들어온 batch 상황에서도 row별 계산이 되는지 확인하는 것이다.
