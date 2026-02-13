
# Humanoid Everyday Visualization

이 프로젝트는 Humanoid 로봇 손(Inspire Hand)에서 수집된 촉각 데이터(Tactile Data)와 카메라 이미지를 동기화하여 시각화하는 도구입니다. 수집된 `json` 포맷의 센서 데이터와 이미지 프레임들을 읽어, 센서의 압력 분포를 2D 히트맵으로 표현하고 이를 원본 비디오와 합쳐 시각화된 `.mp4` 영상을 생성합니다.

## Environments
-   Python 3.10.10 or higher
-   `pip install -r requirements.txt` or <br> `pip install numpy pandas opencv-python matplotlib tqdm`
-   (선택 사항) 결과 영상 압축(`-z` 옵션)을 위해 `ffmpeg`가 시스템에 설치되어 있어야 합니다. 설치되어 있지 않을 경우 `-z` 옵션을 사용할 때 오류가 발생합니다.
    

## 실행 인자 (Arguments)
`main.py`를 실행할 때 사용할 수 있는 인자들은 다음과 같습니다.
-   `-f` | `--folder`: (필수) 데이터가 위치한 최상위 폴더명입니다. 코드는 이 폴더 하위에서 데이터를 재귀적으로 검색합니다. 
- `-e` | `--episode`: (기본값: 0) 시각화할 에피소드 번호입니다. `int` 형으로 입력받습니다.
- `-o` | `--out`: (기본값: `output.mp4`) 생성될 비디오 파일의 경로 및 이름입니다.
- `-z` | `--zip`: (선택) 이 플래그를 사용하면 `ffmpeg`를 이용해 출력 비디오를 압축하여 용량을 줄입니다.
    

**실행 예시:**
```
python main.py \
  --folder click_a_keyboard_key --episode 0 \
  --out click_a_keyboard_key0.mp4 --zip
```

## Directory Structure
이 코드는 데이터가 다음과 같은 구조로 배치되어 있다고 가정하고 `glob`을 통해 파일을 찾습니다. `background.png`와 `sensor id.png`는 시각화 배경 및 매핑 확인용으로 코드와 같은 디렉토리에 위치해야 합니다.
```
project_root/
├── main.py
├── config.py
├── utils.py
├── ShapeGroup.py
├── background.png          # 시각화 배경 이미지
└── {Parent_Path}/          # 임의의 상위 경로
   └── {folder}/            # 실행 인자 -f 로 지정한 폴더
      └── ...
         └── episode_{episode}/
            ├── data.json   # 촉각 센서 데이터
            └── color/      # 카메라 이미지 프레임 폴더
               ├── 0000.jpg
               ├── 0001.jpg
               └── ...
```

## Dataset Format

### 1. Tactile Data (`data.json`)

각 프레임별 로봇 손의 압력 센서 데이터를 담고 있는 JSON 파일입니다. `utils.py`의 `load_tactile` 함수에 따라 다음과 같은 구조를 가져야 합니다.

```
[
  {
    "states": {
      "hand_pressure_state": [
        {
          "sensor_id": 1,
          "usable_readings": [...] 
        },
        ...
      ]
    }
  },
  ...
]
```

-   **sensor_id**: `config.py`의 `SENSOR_MAPPING`에 정의된 센서 ID (Left: 1~9, Right: 10~18)
-   **usable_readings**: 각 센서 패드의 압력 값 배열
    

### 2. Images (`color/*.jpg`)

로봇의 시점에서 촬영된 이미지 파일들입니다. 데이터 JSON의 프레임 순서와 매칭되어 영상으로 합쳐집니다.

## Visualization Logic

1.  **Normalization**: 센서 데이터는 `utils.py`의 로직에 따라 정규화(Normalize) 과정을 거칩니다. 노이즈를 줄이고 변화량을 강조하기 위해 z-score 기반의 처리가 수행됩니다.
    
2.  **Mapping**: `config.py`와 `ShapeGroup.py`를 통해 각 센서 값은 손 모양 이미지(`background.png`) 위의 특정 위치(Grid 또는 Pyramid 형태)에 색상으로 매핑됩니다.
    
3.  **Rendering**:
    -   **상단**: 원본 카메라 이미지 + 실시간 센서 값 그래프(Bar Chart)
    -   **하단**: 좌/우 손의 촉각 분포 시각화 (heatmap)

## Sensor Mapping 관련 참고사항
`config.py`에 저장되어 있는 mapping 값을 수정해 만들어지는 영상의 위치를 수정할 수 있습니다.

- `'left'` 항목은 왼쪽에 위치하며, `'right'` 항목은 오른쪽에 위치합니다.
- 내부 dict은 `{sensor_id}: (group_id, group_idx)`의 형식을 가집니다.
	- `sensor_id`는, `data.json`에 저장되어 있는 `"states", "hand_pressure_state", "sensor_id"` 값을 의미합니다.
	- `group_id`는 sensor에 할당되어 있는 group의 id를 의미합니다. 구체적인 group id는 [dex3-1](https://support.unitree.com/home/en/dex3-1_hand/about_dex3-1) 홈페이지의 내용을 참고 부탁드립니다.
	- `group_idx`는 grid, pyramid 내부 index를 의미합니다. grid의 경우 `(idx1, idx2, idx3, idx4)`의 형식을 가지며, `data.json` 내의 `"states", "hand_pressure_state", "usable_readings"`에 해당하는 값을 순서대로 idx1, idx2, idx3, idx4의 위치에 배치합니다. pyramid의 경우 `(idx1, idx2, idx3)`의 형식을 갖습니다.
