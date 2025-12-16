async function setModel() {
    const modelPath = document.getElementById("modelSelect").value.trim();
    if (!modelPath) {
        alert("請輸入模型路徑");
        return;
    }

    const res = await fetch("/set_model", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_path: modelPath })
    });
    const data = await res.json();
    if (data.status == "ok") {
        alert(`模型設定成功: ${data.model}`);
    } else {
        alert(`模型設定失敗: ${data.message}`);
    }
}

window.onload = async function () {
    const res = await fetch("/get_models");
    const data = await res.json();
    const select = document.getElementById("modelSelect");
    data.models.forEach(m => {
        const opt = document.createElement("option");
        opt.value = `pt/${m}`;  // 完整路徑
        opt.innerText = m;
        select.appendChild(opt);
    });
}

async function upload() {
    const file = document.getElementById("file").files[0];
    if (!file) {
        alert("請先選擇檔案");
        return;
    }

    let form = new FormData();
    form.append("file", file);

    document.getElementById("wait").innerText = "處理中，請稍候...";
    document.getElementById("resultImg").style.display = "none";
    document.getElementById("resultVideo").style.display = "none";
    document.getElementById("result_img_count").innerText = "";
    document.getElementById("result_count_by_class").innerText = "";

    const res = await fetch("/predict", { method: "POST", body: form });
    const data = await res.json();

    // 顯示影片
    if (data.type === "video") {
        document.getElementById("wait").innerText = "";
        document.getElementById("resultVideo").src = data.video_url;
        document.getElementById("resultVideo").style.display = "block";
    }
    // 顯示圖片
    else if (data.type === "image") {
        document.getElementById("wait").innerText = "";
        document.getElementById("resultImg").src = "data:image/jpeg;base64," + data.image;
        document.getElementById("resultImg").style.display = "block";
        document.getElementById("result_img_count").innerText = `偵測到 ${data.count} 個物體`;
    }
    else {
        alert("格式錯誤或模型處理失敗");
    }
}