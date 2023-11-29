function(){
    console.log("custom_inversion_functional.js loaded");
    
    function showModal(imageSrc) {
        var modal = document.getElementById("myModal");
        var modalImg = document.getElementById("img01");
        modal.style.display = "block";
        modalImg.src = imageSrc;

        // Reset scale and transform origin when a new image is shown
        modalImg._zoomScale = 1;  // Initialize a property to keep track of the scale
        updateTransform(modalImg);
    }

    function updateTransform(imgElement) {
        // Combines translate and scale in one transform property
        imgElement.style.transform = `translate(-50%, -50%) scale(${imgElement._zoomScale})`;
    }
    
    document.addEventListener(`click`, e => {
        const origin = e.target.closest(`.gr-image-output`);
        const img = origin && origin.querySelector(`img`);
        if(!img) return;
        
        showModal(img.src);
    });
    
    window.onclick = function(event) {
        var modal = document.getElementById("myModal");
        if (event.target == modal) {
            modal.style.display = "none";
        }
    }

    // Add wheel event listener for zooming
    document.getElementById("img01").addEventListener('wheel', function(event) {
        var scaleIncrement = 0.1;
        var minScale = 0.5;
        var maxScale = 3;

        if(event.deltaY < 0) { // Zoom in
            this._zoomScale = Math.min(this._zoomScale + scaleIncrement, maxScale);
        } else { // Zoom out
            this._zoomScale = Math.max(this._zoomScale - scaleIncrement, minScale);
        }

        updateTransform(this);
        event.preventDefault(); // Prevent page scrolling
    });

    function toogleHistory(status){
        // let checkboxes_container = document.querySelectorAll('.history-checkbox');
        // checkboxes_container.forEach(function(checkbox_container) {
        //     let checkbox = checkbox_container.querySelector('input');
        //     checkbox.checked = status;
        // });
        let checkboxes_container = document.querySelectorAll('.history-checkbox');
        checkboxes_container.forEach(function(checkbox_container) {
            //if not status, then click
            if(checkbox_container.querySelector('input').checked == status) return;
            let checkbox = checkbox_container.querySelector('label');
            checkbox.click();
        });
    }
    document.getElementById('activate_button').addEventListener('click', function() {
        console.log("Activate button clicked");
        toogleHistory(true);
    });
    document.getElementById('deactivate_button').addEventListener('click', function() {
        console.log("Deactivate button clicked");
        toogleHistory(false);
    });
}