document.addEventListener('DOMContentLoaded', function() {
    const sections = document.querySelectorAll('section[id]');
    const navLinks = document.querySelectorAll('nav ul li a');

    // Add click event listener to each navigation link
    navLinks.forEach(link => {
        link.addEventListener('click', function(event) {
            event.preventDefault();
            const targetId = this.getAttribute('href').substring(1); // Get the target section ID
            scrollToSection(targetId); // Scroll to the target section
        });
    });

    // Function to scroll to the target section
    function scrollToSection(id) {
        const targetSection = document.getElementById(id);
        if (targetSection) {
            // Hide all sections
            sections.forEach(section => {
                section.style.display = 'none';
            });

            // Display the target section
            targetSection.style.display = 'block';

            // Optionally, you can scroll to the target section if needed
            targetSection.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    }

    // Initially show the first section or the section based on the URL hash
    const initialSectionId = window.location.hash.substring(1);
    if (initialSectionId) {
        scrollToSection(initialSectionId);
    } else {
        scrollToSection('experience'); // Default section to show
    }
});
