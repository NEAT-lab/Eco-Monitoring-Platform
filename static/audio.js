async function uploadAudio() {
    const fileInput = document.getElementById('audioFile');
    const resultChart = document.getElementById('resultChart');
    const waitDiv = document.getElementById('wait');

    // 1. 基本檢查
    if (fileInput.files.length === 0) {
        alert("請選擇音訊或影片檔案！");
        return;
    }

    const file = fileInput.files[0];
    
    // 2. 前端限制：最大 15MB
    const MAX_SIZE = 15 * 1024 * 1024; 
    if (file.size > MAX_SIZE) {
        alert("檔案過大，請選擇 15MB 以下的檔案。");
        return;
    }

    // 3. 準備發送
    const formData = new FormData();
    formData.append('file', file);

    // 更新 UI 狀態
    waitDiv.innerHTML = `<p style="color: #007bff;">正在分析「${file.name}」...請稍候</p>`;
    resultChart.style.display = 'none';
    resultChart.src = "";

    try {
        const response = await fetch('/api/audio', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`伺服器回應異常: ${response.status}`);
        }

        const data = await response.json();

        if (data.type === 'image') {
            // 成功：渲染 Base64 圖表
            resultChart.src = "data:image/png;base64," + data.image;
            resultChart.style.display = 'block';
            waitDiv.innerHTML = `<p style="color: green;">分析完成！</p>`;
        } else {
            // 錯誤
            waitDiv.innerHTML = `<p style="color: red;">錯誤：${data.message}</p>`;
        }

    } catch (error) {
        console.error("Analysis failed:", error);
        waitDiv.innerHTML = `<p style="color: red;">發生連線錯誤或系統異常</p>`;
    }
}