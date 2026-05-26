# ReLU 학습 문서

## 1. 핵심 역할

`ReLU(Rectified Linear Unit)`는 은닉층에서 사용하는 활성화 함수다.

입력값이 양수이면 그대로 통과시키고, 0 이하이면 0으로 막는다.

```text
ReLU(x) = max(0, x)
```

신경망에서 `Affine`만 여러 번 쌓으면 결국 하나의 선형 변환과 비슷해진다. `ReLU`는 중간에 비선형성을 넣어서 모델이 더 복잡한 패턴을 학습할 수 있게 만든다.

MNIST 모델에서는 다음 위치에서 사용된다.

```text
Affine -> BatchNorm -> ReLU -> Dropout
```

## 배경: 왜 ReLU가 나왔는가

초기 신경망에서는 `sigmoid`나 `tanh` 같은 활성화 함수를 많이 사용했다.

```text
sigmoid(x) = 1 / (1 + exp(-x))
tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
```

이 함수들은 입력을 부드러운 곡선으로 바꿔 주지만, 입력값의 절댓값이 커지면 기울기가 거의 0에 가까워진다.

sigmoid의 미분은 다음과 같다.

```text
sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
```

`sigmoid(x)`는 0과 1 사이 값이다. 입력이 아주 크면 `sigmoid(x)`는 1에 가까워지고, 입력이 아주 작으면 0에 가까워진다.

```text
x가 매우 큼: sigmoid(x) ≈ 1 -> sigmoid'(x) ≈ 1 * 0 = 0
x가 매우 작음: sigmoid(x) ≈ 0 -> sigmoid'(x) ≈ 0 * 1 = 0
```

역전파는 여러 층의 미분값을 계속 곱해서 앞쪽 층으로 gradient를 전달한다. 그런데 각 층에서 0에 가까운 값이 계속 곱해지면 앞쪽 층의 gradient가 거의 사라진다. 이것을 `vanishing gradient` 문제라고 한다.

ReLU는 양수 구간에서 기울기가 항상 1이다.

```text
ReLU(x) = max(0, x)

x > 0 이면 ReLU'(x) = 1
x < 0 이면 ReLU'(x) = 0
```

양수 구간에서는 gradient가 줄어들지 않고 그대로 전달된다. 그래서 깊은 신경망에서 sigmoid보다 학습이 잘 되는 경우가 많다.

물론 ReLU도 단점이 있다. 입력이 계속 음수로 들어오면 출력도 0, gradient도 0이 되어 해당 뉴런이 학습되지 않을 수 있다. 이것을 `dying ReLU`라고 부른다.

## 공식과 미분 유도

ReLU는 구간별 함수로 볼 수 있다.

```text
          { x,  x > 0
ReLU(x) = {
          { 0,  x <= 0
```

각 구간에서 미분하면 다음과 같다.

```text
x > 0:  d(x)/dx = 1
x < 0:  d(0)/dx = 0
```

`x = 0`에서는 왼쪽 기울기와 오른쪽 기울기가 다르다.

```text
왼쪽 기울기: 0
오른쪽 기울기: 1
```

따라서 수학적으로는 0에서 미분 불가능하다. 하지만 실제 구현에서는 보통 0으로 처리한다. 현재 코드도 `x > 0`만 통과시키므로 `x == 0`인 위치는 gradient가 0이 된다.

역전파에서 다음 층으로부터 `dout = dL/dy`가 왔다고 하자. 여기서 `y = ReLU(x)`다. 체인 룰을 적용하면 다음과 같다.

```text
dL/dx = dL/dy * dy/dx
```

`dy/dx`는 양수 구간에서 1, 0 이하 구간에서 0이다.

```text
dL/dx = dout * 1  if x > 0
dL/dx = dout * 0  if x <= 0
```

이것이 코드의 `dout * self.mask`로 이어진다.

## 2. Forward 흐름

현재 구현 파일은 `src/activations.py`다.

```python
self.mask = (x > 0)
out = x * self.mask
```

흐름은 다음과 같다.

1. 입력 `x`에서 양수인 위치를 `True`, 0 이하인 위치를 `False`로 표시한다.
2. 이 값을 `self.mask`에 저장한다.
3. `x * self.mask`를 계산한다.
4. `True` 위치는 원래 값이 남고, `False` 위치는 0이 된다.

예를 들어 입력이 다음과 같다고 하자.

```text
x = [-2, 0, 3]
mask = [False, False, True]
out = [0, 0, 3]
```

Forward에서 `mask`를 저장하는 이유는 backward 때 같은 위치에 gradient를 흘릴지 말지 결정해야 하기 때문이다.

## 3. Backward 흐름

현재 구현은 다음과 같다.

```python
return dout * self.mask
```

`dout`은 다음 층에서 넘어온 gradient다. `ReLU`는 forward 때 0 이하였던 위치를 막았으므로, backward에서도 그 위치로는 gradient가 흐르면 안 된다.

```text
x <= 0 이었던 위치: gradient = 0
x > 0 이었던 위치: gradient 그대로 통과
```

예를 들어 forward 때 입력이 다음과 같았다면:

```text
x = [-2, 0, 3]
mask = [False, False, True]
```

다음 층에서 온 gradient가 `[10, 20, 30]`이어도 backward 결과는 다음과 같다.

```text
dx = [0, 0, 30]
```

즉, forward에서 죽은 뉴런은 backward에서도 gradient가 전달되지 않는다.

## 4. 구현 포인트

가장 중요한 점은 forward에서 만든 `mask`를 backward에서 다시 사용한다는 것이다.

```python
self.mask = (x > 0)
```

이 코드는 단순히 출력 계산을 위한 것이 아니라, 역전파 때 gradient 흐름을 제어하기 위한 기록이다.

`x * self.mask`에서 `True`는 1처럼, `False`는 0처럼 동작한다. NumPy에서는 boolean 배열과 숫자 배열을 곱할 수 있기 때문에 간단하게 구현할 수 있다.

## 5. 자주 헷갈리는 점

`x == 0`일 때는 현재 구현에서 `False`가 된다.

```python
self.mask = (x > 0)
```

따라서 0인 위치도 gradient가 0으로 막힌다. ReLU는 0에서 미분이 엄밀히 정의되지 않지만, 구현에서는 보통 0으로 처리한다.

또 하나 중요한 점은 `mask`의 shape가 입력 `x`와 같아야 한다는 것이다. 그래야 forward와 backward에서 위치별로 같은 뉴런을 제어할 수 있다.

## 6. 테스트 관점

`tests/test_relu.py`에서는 보통 다음 동작을 확인한다.

- forward에서 음수와 0은 0이 되는가
- forward에서 양수는 그대로 유지되는가
- backward에서 forward 때 0 이하였던 위치의 gradient가 0이 되는가
- 출력 shape가 입력 shape와 같은가

`ReLU` 테스트를 통과한다는 것은 단순히 값만 맞는 것이 아니라, forward에서 저장한 `mask`가 backward까지 올바르게 이어진다는 뜻이다.
