async function updateData() {
    const res = await fetch("/api/data");
    const data = await res.json();

    temp.innerText = data.temperature ?? "--";
    hum.innerText  = data.humidity ?? "--";
    time.innerText = data.timestamp ?? "--";
}

setInterval(updateData, 10000);
updateData();
