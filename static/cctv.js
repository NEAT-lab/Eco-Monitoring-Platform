async function updateData() {
    const res = await fetch("/api/data");
    const data = await res.json();

    temp.innerText = data.temperature ?? "--";
    hum.innerText = data.humidity ?? "--";
    time.innerText = data.timestamp ?? "--";
}

// ========== YOLO 識別結果更新 ==========
async function updateDetectionImage(cameraId) {
    try {
        const statusElement = document.getElementById(`detection${cameraId === 'cam1' ? '1' : '2'}_status`);
        const imgElement = document.getElementById(`detection_image_cam${cameraId === 'cam1' ? '1' : '2'}`);
        const placeholderElement = document.getElementById(`detection${cameraId === 'cam1' ? '1' : '2'}_placeholder`);
        
        // 獲取識別信息
        const infoResponse = await fetch(`/api/detection_info/${cameraId}`);
        const info = await infoResponse.json();
        
        if (info.has_result) {
            const lastDetection = new Date(info.last_detection);
            const nextIn = Math.ceil(info.next_detection_in);
            statusElement.textContent = `最後識別時間: ${lastDetection.toLocaleTimeString('zh-TW')} (下次識別: ${nextIn}秒後)`;
            
            // 獲取識別圖像
            const imgResponse = await fetch(`/api/detection_image/${cameraId}`);
            if (imgResponse.ok) {
                const blob = await imgResponse.blob();
                const url = URL.createObjectURL(blob);
                imgElement.src = url;
                imgElement.style.display = 'block';
                placeholderElement.style.display = 'none';
            }
        } else {
            statusElement.textContent = '等待第一次識別結果...';
        }
    } catch (error) {
        console.error(`更新${cameraId}識別結果失敗:`, error);
    }
}

// 每10秒更新一次識別結果（間隔不要太短，因為識別是每60秒執行一次）
setInterval(() => {
    updateDetectionImage('cam1');
    updateDetectionImage('cam2');
}, 10000);

// 初始化時立即更新一次
updateDetectionImage('cam1');
updateDetectionImage('cam2');

// ========== 手動識別功能 ==========
async function manualDetect(cameraId) {
    const buttonId = `manualDetect_cam${cameraId === 'cam1' ? '1' : '2'}`;
    const statusId = `detection${cameraId === 'cam1' ? '1' : '2'}_status`;
    const imgId = `detection_image_cam${cameraId === 'cam1' ? '1' : '2'}`;
    const placeholderId = `detection${cameraId === 'cam1' ? '1' : '2'}_placeholder`;
    
    const button = document.getElementById(buttonId);
    const statusElement = document.getElementById(statusId);
    const originalText = button.textContent;
    
    try {
        button.disabled = true;
        button.textContent = '識別中...';
        statusElement.textContent = '正在執行識別...';
        
        const response = await fetch(`/api/manual_detect/${cameraId}`, {
            method: 'POST'
        });
        
        if (!response.ok) {
            const error = await response.json();
            statusElement.textContent = `識別失敗: ${error.error}`;
            return;
        }
        
        const data = await response.json();
        
        if (data.status === 'success') {
            // 更新圖像
            const imgElement = document.getElementById(imgId);
            const placeholderElement = document.getElementById(placeholderId);
            
            imgElement.src = 'data:image/jpeg;base64,' + data.image;
            imgElement.style.display = 'block';
            placeholderElement.style.display = 'none';
            
            const detectionTime = new Date(data.timestamp);
            statusElement.textContent = `識別完成: 發現 ${data.bird_count} 隻鳥 | ${detectionTime.toLocaleTimeString('zh-TW')}`;
        }
        
    } catch (error) {
        statusElement.textContent = `錯誤: ${error.message}`;
    } finally {
        button.disabled = false;
        button.textContent = originalText;
    }
}

// 綁定手動識別按鈕事件
const manualDetectBtn1 = document.getElementById('manualDetect_cam1');
const manualDetectBtn2 = document.getElementById('manualDetect_cam2');

if (manualDetectBtn1) {
    manualDetectBtn1.addEventListener('click', () => manualDetect('cam1'));
}
if (manualDetectBtn2) {
    manualDetectBtn2.addEventListener('click', () => manualDetect('cam2'));
}

document.getElementById("rtsp_test").src = "/video_feed/cam1?t=" + Date.now();
document.getElementById("rtsp_test_2").src = "/video_feed/cam2?t=" + Date.now();

fetch('/api/get_zoom')
  .then(res => res.json())
  .then(data => {
      document.getElementById('zoomDisplay').innerText = data.zoom;
      document.getElementById('zoomSlider').value = data.zoom;
  });

fetch('/api/get_zoom_2')
  .then(res => res.json())
  .then(data => {
      document.getElementById('zoomDisplay_2').innerText = data.zoom;
      document.getElementById('zoomSlider_2').value = data.zoom;
  });

// ========== 左攝影機控制 ==========
const zoomSlider = document.getElementById('zoomSlider');
const zoomDisplay = document.getElementById('zoomDisplay');
const zoomStatus = document.getElementById('zoomStatus');

let debounceTimer;

zoomSlider.addEventListener('input', (e) => {
    const zoomValue = parseInt(e.target.value);
    zoomDisplay.textContent = zoomValue;

    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
        sendZoomCommand(zoomValue);
    }, 1000);
});

async function sendZoomCommand(zoomValue) {
    try {
        zoomStatus.textContent = '發送中...';

        const response = await fetch('/api/zoom', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ zoom: zoomValue })
        });

        if (response.ok) {
            zoomStatus.textContent = '成功';
        } else {
            zoomStatus.textContent = '錯誤';
        }
    } catch (error) {
        zoomStatus.textContent = '錯誤: ' + error.message;
    }
}


// ========== 右攝影機控制 ==========
// ==== 右攝影機方向控制 ====
const directionBtns = {
    up: document.getElementById('upBtn'),
    down: document.getElementById('downBtn'),
    left: document.getElementById('leftBtn'),
    right: document.getElementById('rightBtn')
};
const directionStatus = document.getElementById('directionStatus');

// directionBtns.up.addEventListener('click', () => sendDirection('up'));
// directionBtns.down.addEventListener('click', () => sendDirection('down'));
// directionBtns.left.addEventListener('click', () => sendDirection('left'));
// directionBtns.right.addEventListener('click', () => sendDirection('right'));

// async function sendDirection(direction) {
//     try {
//         directionStatus.textContent = '發送中...';

//         const response = await fetch('/api/direction_2', {
//             method: 'POST',
//             headers: {
//                 'Content-Type': 'application/json'
//             },
//             body: JSON.stringify({ direction: direction })
//         });

//         if (response.ok) {
//             directionStatus.textContent = '成功';
//         } else {
//             directionStatus.textContent = '錯誤';
//         }
//     } catch (error) {
//         directionStatus.textContent = '錯誤: ' + error.message;
//     }
// }

// ==== 右攝影機變焦控制 ====
const zoomSlider_2 = document.getElementById('zoomSlider_2');
const zoomDisplay_2 = document.getElementById('zoomDisplay_2');
const zoomStatus_2 = document.getElementById('zoomStatus_2');

let debounceTimer_2;

zoomSlider_2.addEventListener('input', (e) => {
    const zoomValue_2 = parseInt(e.target.value);
    zoomDisplay_2.textContent = zoomValue_2;

    clearTimeout(debounceTimer_2);
    debounceTimer_2 = setTimeout(() => {
        sendZoomCommand_2(zoomValue_2);
    }, 1000);
});

async function sendZoomCommand_2(zoomValue) {
    try {
        zoomStatus_2.textContent = '發送中...';

        const response = await fetch('/api/zoom_2', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ zoom: zoomValue })
        });

        if (response.ok) {
            zoomStatus_2.textContent = '成功';
        } else {
            zoomStatus_2.textContent = '錯誤';
        }
    } catch (error) {
        zoomStatus_2.textContent = '錯誤: ' + error.message;
    }
}

document.addEventListener("DOMContentLoaded", function () {
    // 1. 初始化變數與 DOM 元素
    const ctx = document.getElementById('birdChart').getContext('2d');
    const datePicker = document.getElementById('datePicker');
    const btnPrev = document.getElementById('btnPrev');
    const btnNext = document.getElementById('btnNext');
    const btnToday = document.getElementById('btnToday');

    // 取得圖片相關元素
    const recordImage = document.getElementById('recordImage');
    const imageTitle = document.getElementById('imageTitle');
    const noImageText = document.getElementById('noImageText');

    // 設定初始日期為今天
    let currentDate = new Date();

    // 用來記錄上一次的狀態，避免重複刷新圖片
    let lastMaxHour = -1;
    let lastMaxCount = -1;
    let lastDateStr = "";

    // 用來標記使用者是否正在手動查看某張圖
    let isManualSelection = false; 

    // 格式化日期為 YYYY-MM-DD
    function formatDate(date) {
        const year = date.getFullYear();
        const month = String(date.getMonth() + 1).padStart(2, '0');
        const day = String(date.getDate()).padStart(2, '0');
        return `${year}-${month}-${day}`;
    }

    function updateDatePicker() {
        datePicker.value = formatDate(currentDate);
    }

    updateDatePicker();

    function loadImage(hour, count) {
        const dateStr = formatDate(currentDate);
        // 檔名規則: bird_YYYY-MM-DD_H.jpg (注意 hour 是整數，沒有補0)
        // 加上 ?t=... 是為了防止瀏覽器快取舊照片
        const imagePath = `/static/captures/bird_${dateStr}_${hour}.jpg?t=${new Date().getTime()}`;

        imageTitle.innerText = `${dateStr} ${hour}:00 - 最大數量: ${count} 隻`;

        // 預載圖片，等載入完成後再切換，避免破圖或閃爍
        const tempImg = new Image();
        tempImg.src = imagePath;

        tempImg.onload = function () {
            recordImage.src = imagePath;
            recordImage.style.display = 'block';
            noImageText.style.display = 'none';
        };

        tempImg.onerror = function () {
            // 只有在真的找不到圖時才隱藏
            // recordImage.style.display = 'none'; // 選擇性：你可以保留舊圖或隱藏
            noImageText.style.display = 'block';
            noImageText.innerText = `( ${hour}:00 尚無紀錄照片 )`;
        };
    }

    // 2. 初始化 Chart.js (設定為長條圖)
    const hourLabels = Array.from({ length: 24 }, (_, i) => `${i}:00`);
    const birdChart = new Chart(ctx, {
        type: 'bar', // [修改] 這裡改成 'bar' 即為長條圖
        data: {
            labels: hourLabels,
            datasets: [{
                label: '每小時最大鳥類數量',
                data: [],
                // 長條圖樣式設定
                backgroundColor: 'rgba(54, 162, 235, 0.6)', // 長條內部的顏色 (半透明藍色)
                borderColor: 'rgba(54, 162, 235, 1)',       // 邊框顏色 (深藍色)
                borderWidth: 1,                              // 邊框寬度
                borderRadius: 4,                             // [選用] 讓長條頂端圓角化
                barPercentage: 0.8                           // [選用] 控制長條寬度 (0.1 ~ 1.0)
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            onClick: (e) => {
                const points = birdChart.getElementsAtEventForMode(e, 'nearest', { intersect: true }, true);
                if (points.length) {
                    const index = points[0].index; // 取得點擊的索引 (即小時 0-23)
                    const count = birdChart.data.datasets[0].data[index]; // 取得該小時的數量
                    // 呼叫載入照片函式
                    loadImage(index, count);
                    isManualSelection = true;
                }
            },
            plugins: {
                title: { display: true, text: '載入中...', font: { size: 16 } },
                tooltip: { mode: 'index', intersect: false },
                legend: { display: true, position: 'top' }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: { display: true, text: '數量 (隻)' },
                    ticks: { stepSize: 1 }
                },
                x: {
                    title: { display: true, text: '時間 (小時)' },
                    grid: { display: false } // [選用] 隱藏 X 軸網格讓長條更乾淨
                }
            }
        }
    });

    // 3. 抓取數據並更新圖表 (邏輯不變)
    async function fetchDailyData() {
        const dateStr = formatDate(currentDate);
        birdChart.options.plugins.title.text = `${dateStr} 鳥類活動統計`;

        try {
            const response = await fetch(`/api/daily_stats?date=${dateStr}`);
            const result = await response.json();

            if (result.data) {
                birdChart.data.datasets[0].data = result.data;
                birdChart.update('none'); // 'none' 參數可以讓圖表更新時不要有太誇張的動畫

                // [新增] 自動找出當天數量最多的那個小時並顯示照片
                const maxCount = Math.max(...result.data);
                if (maxCount > 0) {
                    // 找到最大值的索引 (小時)
                    const maxHour = result.data.indexOf(maxCount);
                    // 自動載入那張照片
                    // 只有在「非手動模式」下，才允許自動切換圖片
                    if (!isManualSelection && (dateStr !== lastDateStr || maxHour !== lastMaxHour || maxCount > lastMaxCount)) {
                        console.log("偵測到數據更新，刷新圖片...");
                        loadImage(maxHour, maxCount);

                        // 更新狀態紀錄
                        lastMaxHour = maxHour;
                        lastMaxCount = maxCount;
                        lastDateStr = dateStr;
                    }
                } else {
                    // 如果當天完全沒數據
                    if (lastMaxCount !== 0) {
                        recordImage.style.display = 'none';
                        noImageText.style.display = 'block';
                        noImageText.innerText = "( 今日尚無數據 )";
                        lastMaxCount = 0;
                    }
                }
            }
        } catch (error) {
            console.error("獲取數據失敗:", error);
        }
    }

    // 4. 事件監聽器 (邏輯不變)
    btnPrev.addEventListener('click', () => {
        currentDate.setDate(currentDate.getDate() - 1);
        lastDateStr = ""; // 重置日期紀錄，確保圖片會更新
        isManualSelection = false; // 切換日期後，重置為自動模式
        updateDatePicker();
        fetchDailyData();
    });

    btnNext.addEventListener('click', () => {
        currentDate.setDate(currentDate.getDate() + 1);
        lastDateStr = "";
        isManualSelection = false;
        updateDatePicker();
        fetchDailyData();
    });

    btnToday.addEventListener('click', () => {
        currentDate = new Date();
        lastDateStr = "";
        isManualSelection = false;
        updateDatePicker();
        fetchDailyData();
    });

    datePicker.addEventListener('change', (e) => {
        if (e.target.value) {
            currentDate = new Date(e.target.value);
            lastDateStr = "";
            isManualSelection = false;  
            fetchDailyData();
        }
    });

    // 啟動
    fetchDailyData();

    // 自動刷新 (僅限今天)
    setInterval(() => {
        const todayStr = formatDate(new Date());
        const currentStr = formatDate(currentDate);
        if (todayStr === currentStr) {
            fetchDailyData();
        }
    }, 10000);
});

setInterval(updateData, 60000);
updateData();
