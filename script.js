document.addEventListener('DOMContentLoaded', function() {
    const form = document.querySelector('form');
    form.addEventListener('submit', function(event) {
        event.preventDefault();
        alert('Message sent!');
        form.reset();
    });
});

function toggleDescription(card) {
    // Find the description element within the clicked card
    var description = card.nextElementSibling;

    // Toggle the visibility of the description
    if (description.style.display === "block") {
        description.style.display = "none";
    } else {
        description.style.display = "block";
    }
}