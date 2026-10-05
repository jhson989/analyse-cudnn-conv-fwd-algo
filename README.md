# cuDNN Convolution Forward Algorithm 분석

동일한 Convolution 레이어 설정에 대해 cuDNN이 제공하는 Forward 알고리즘(`cudnnConvolutionFwdAlgo_t`)별 **수행 시간**과 **workspace 메모리 사용량**을 측정/비교하는 벤치마크입니다.

## 설명
https://computing-jhson.tistory.com/46

## 측정 대상 알고리즘
| 알고리즘 | 비고 |
|---|---|
| `IMPLICIT_GEMM` | |
| `IMPLICIT_PRECOMP_GEMM` | |
| `GEMM` | |
| `DIRECT` | cuDNN 미지원으로 제외 |
| `FFT` | |
| `FFT_TILING` | |
| `WINOGRAD` | 3x3 필터에서만 측정 |
| `WINOGRAD_NONFUSED` | 3x3, 5x5 필터에서만 측정 |

각 알고리즘은 `TEST_ITERATION`(기본 50)회 반복 실행되며, CUDA Event로 전체 시간을 측정합니다.

## 프로젝트 구조
```
.
├── main.cu              # 레이어 설정 파싱, 데이터/디스크립터 생성, 알고리즘별 벤치마크 호출
├── conv_fwd_algo.cuh    # 공용 선언, 에러 체크 매크로, TEST_ITERATION
├── src
│   ├── conv_fwd_algo.cu # launch_conv_fwd(): workspace 할당 및 cudnnConvolutionForward 측정
│   └── debug.cu         # 에러 처리, 알고리즘 이름 테이블, 결과 출력
├── Makefile
└── script.sh            # 필터 크기(1/3/5/7/9)별 일괄 실행 스크립트
```

## 요구 사항
- NVIDIA GPU
- CUDA Toolkit (`/usr/local/cuda/bin/nvcc`)
- cuDNN

## 빌드 및 실행
```bash
make        # conv_fwd.out 빌드
make run    # 기본 설정으로 실행
make clean  # 실행 파일 삭제
```

### 레이어 설정 지정
```bash
./conv_fwd.out BATCH IC IH IW OC FH FW PH PW SH SW
# 예) ./conv_fwd.out 1024 3 128 128 6 3 3 1 1 1 1
```
| 인자 | 의미 |
|---|---|
| `BATCH` | 배치 크기 |
| `IC`, `IH`, `IW` | 입력 채널 / 높이 / 너비 |
| `OC` | 출력 채널 (필터 개수) |
| `FH`, `FW` | 필터 높이 / 너비 |
| `PH`, `PW` | 패딩 (높이 / 너비) |
| `SH`, `SW` | 스트라이드 (높이 / 너비) |

인자를 주지 않으면 기본값(`BATCH=128, 3x256x256 입력, 3x3 필터 3개, pad 0, stride 1`)으로 실행되며, dilation은 항상 1입니다.

### 일괄 실행
```bash
sh script.sh
```
배치 1024, 3→6 채널, 128x128 입력에 대해 필터 크기 1/3/5/7/9 (same padding, stride 1)를 순서대로 측정합니다.

## 출력 예시 형식
```
INPUT : [1024,3,128,128]
FILTER : [6,3,3,3]
LAYER CONFIG : PAD[1,1], STRIDE[1,1], DILATION[1,1]
[GEMM]
 : <시간> s , <workspace 크기> GB
...
```
