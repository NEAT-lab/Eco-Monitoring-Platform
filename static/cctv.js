async function updateData() {
    const res = await fetch("/api/data");
    const data = await res.json();

    temp.innerText = data.temperature ?? "--";
    hum.innerText  = data.humidity ?? "--";
    time.innerText = data.timestamp ?? "--";
}

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

directionBtns.up.addEventListener('click', () => sendDirection('up'));
directionBtns.down.addEventListener('click', () => sendDirection('down'));
directionBtns.left.addEventListener('click', () => sendDirection('left'));
directionBtns.right.addEventListener('click', () => sendDirection('right'));

async function sendDirection(direction) {
    try {
        directionStatus.textContent = '發送中...';

        const response = await fetch('/api/direction_2', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ direction: direction })
        });

        if (response.ok) {
            directionStatus.textContent = '成功';
        } else {
            directionStatus.textContent = '錯誤';
        }
    } catch (error) {
        directionStatus.textContent = '錯誤: ' + error.message;
    }
}

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

setInterval(updateData, 10000);
updateData();
