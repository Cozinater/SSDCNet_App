const inputFile = $(".input-file");
const previewContainer = $(".image-preview");
const previewImage = $(".image-preview__image");
const previewDefaultText = $(".image-preview__default-text");
const runModelButton = $(".button");
const result = $(".result");
const runningModel = $(".inline");
const loader = $(".loader");
const openWebcam = $(".open_webcam");
const closeWebcam = $(".close_webcam");
const displayWebcam = $(".display_webcam");
const webcamDefaultText = $(".webcam-preview__default-text");
const openVideo = $(".open_video");
const closeVideo = $(".close_video")

closeWebcam.click(function() {
    displayWebcam.addClass("display-none");
    webcamDefaultText.text("Webcam is off...");
    displayWebcam.attr("src", "");
    webcamDefaultText.removeClass("display-none")
});

closeVideo.click(function() {
    displayWebcam.addClass("display-none");
    webcamDefaultText.text("Video is off...");
    displayWebcam.attr("src", "");
    webcamDefaultText.removeClass("display-none")
});

openVideo.on("click", function() {
    displayWebcam.attr("src", "/video_feed");
    webcamDefaultText.text("Opening Video...");
    setTimeout(function () {
        webcamDefaultText.text("Video is off...");
        webcamDefaultText.addClass("display-none")
        displayWebcam.removeClass("display-none");
    }, 4000);

});

openWebcam.on("click", function() {
    displayWebcam.attr("src", "/webcam_feed");
    webcamDefaultText.text("Opening Webcam...");
    setTimeout(function () {
        webcamDefaultText.text("Webcam is off...");
        webcamDefaultText.addClass("display-none")
        displayWebcam.removeClass("display-none");
    }, 10000);

});

inputFile.on("click", function() {
    const file = this.files[0];
    console.log(file);

    if (file) {
        const reader = new FileReader();
        previewDefaultText.addClass("display-none");
        previewImage.addClass("display-block");
        runModelButton.removeClass("display-none")

        reader.addEventListener("load", function() {
            console.log(this);
            previewImage.attr("src", this.result);
        });

        reader.readAsDataURL(file);
    } else {
        previewDefaultText.removeClass("display-none");
        previewImage.removeClass("display-block");
        //previewImage.attr("src", "");
        runModelButton.addClass("display-none")
    }
});

runModelButton.on("click", function() {
    runningModel.removeClass("display-none");
    runningModel.addClass("display-flex");
    setTimeout(function () {
        result.removeClass("display-none");
        loader.addClass("display-none")
    }, 5000);
});