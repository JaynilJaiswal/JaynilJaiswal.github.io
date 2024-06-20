function toggleDetails(id) {
    var details = document.getElementById(id);
    if (details.style.display === "none" || details.style.display === "") {
        details.style.display = "block";
    } else {
        details.style.display = "none";
    }
}

// Add interactivity to form submission (optional)
document.querySelector('form').addEventListener('submit', function(event) {
    event.preventDefault();
    alert('Thank you for reaching out! I will get back to you soon.');
});