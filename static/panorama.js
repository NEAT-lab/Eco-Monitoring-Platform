let modelPath = "";

async function setModel() {
    modelPath = document.getElementById("modelSelect").value.trim();
    if (!modelPath) {
        alert("請輸入模型路徑");
        return;
    }
    alert(`已選擇模型: ${modelPath}`);
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
    modelPath = "pt/yolov8n.pt";  // 預設模型
    select.value = modelPath;
}

async function upload() {
    const file = document.getElementById("file").files[0];
    if (!file) {
        alert("請先選擇檔案");
        return;
    }

    let form = new FormData();
    form.append("file", file);
    form.append("model_path", modelPath);

    document.getElementById("wait").innerText = "處理中，請稍候...";
    document.getElementById("resultImg").style.display = "none";
    document.getElementById("result_img_count").innerText = "";
    document.getElementById("result_count_by_class").innerText = "";

    try {
        const res = await fetch("/api/panorama", { method: "POST", body: form });
        const data = await res.json();

        // 根據後端回傳的格式判斷
        if (data.type === "image") {
            document.getElementById("wait").innerText = "";
            document.getElementById("resultImg").src = "data:image/jpeg;base64," + data.image;
            document.getElementById("resultImg").style.display = "block";
            document.getElementById("result_img_count").innerText = `偵測到 ${data.total_count} 個物體`;

            let class_text = "各類別數量: ";
            for (const [cls, num] of Object.entries(data.count_by_class)) {
                class_text += `${cls}: ${num} `;
            }
            document.getElementById("result_count_by_class").innerText = class_text;
        } else {
            document.getElementById("wait").innerText = "";
            alert(`處理失敗: ${data.message}`);
        }
    } catch (error) {
        document.getElementById("wait").innerText = "";
        alert(`發生錯誤: ${error.message}`);
        console.error(error);
    }
}