function startWebcam() {
    const videoElement = document.getElementById('video');
    const constraints = {
        video: { facingMode: "user" }
    };

    navigator.mediaDevices.getUserMedia(constraints)
        .then(function(stream) {
            videoElement.srcObject = stream;
        })
        .catch(function(error) {
            console.error('Error accessing webcam:', error);
        });
}
