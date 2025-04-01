// Funcionalidad principal para la interfaz web del curso
document.addEventListener('DOMContentLoaded', function() {
    // Menú móvil
    const menuToggle = document.querySelector('.menu-toggle');
    const nav = document.querySelector('nav');
    
    if (menuToggle) {
        menuToggle.addEventListener('click', function() {
            nav.classList.toggle('active');
        });
    }
    
    // Navegación suave
    const navLinks = document.querySelectorAll('nav a, .footer-links a');
    
    navLinks.forEach(link => {
        if (link.hash && document.querySelector(link.hash)) {
            link.addEventListener('click', function(e) {
                e.preventDefault();
                
                // Cerrar menú móvil si está abierto
                if (nav.classList.contains('active')) {
                    nav.classList.remove('active');
                }
                
                const targetId = this.getAttribute('href');
                const targetElement = document.querySelector(targetId);
                
                if (targetElement) {
                    window.scrollTo({
                        top: targetElement.offsetTop - 80,
                        behavior: 'smooth'
                    });
                    
                    // Actualizar URL
                    history.pushState(null, null, targetId);
                }
            });
        }
    });
    
    // Navegación activa en scroll
    const sections = document.querySelectorAll('section[id]');
    
    function highlightNavOnScroll() {
        const scrollPosition = window.scrollY;
        
        sections.forEach(section => {
            const sectionTop = section.offsetTop - 100;
            const sectionHeight = section.offsetHeight;
            const sectionId = section.getAttribute('id');
            
            if (scrollPosition >= sectionTop && scrollPosition < sectionTop + sectionHeight) {
                document.querySelector('nav a[href="#' + sectionId + '"]').classList.add('active');
            } else {
                document.querySelector('nav a[href="#' + sectionId + '"]').classList.remove('active');
            }
        });
    }
    
    window.addEventListener('scroll', highlightNavOnScroll);
    
    // Animaciones al hacer scroll
    const animateElements = document.querySelectorAll('.feature-card, .module-card, .interactive-card, .resource-card, .step');
    
    function checkIfInView() {
        const windowHeight = window.innerHeight;
        const windowTopPosition = window.scrollY;
        const windowBottomPosition = windowTopPosition + windowHeight;
        
        animateElements.forEach(element => {
            const elementHeight = element.offsetHeight;
            const elementTopPosition = element.offsetTop;
            const elementBottomPosition = elementTopPosition + elementHeight;
            
            // Verificar si el elemento está en la vista
            if ((elementBottomPosition >= windowTopPosition) && (elementTopPosition <= windowBottomPosition)) {
                element.classList.add('animate');
            }
        });
    }
    
    // Agregar clase para CSS
    animateElements.forEach(element => {
        element.classList.add('animate-on-scroll');
    });
    
    window.addEventListener('scroll', checkIfInView);
    window.addEventListener('resize', checkIfInView);
    window.addEventListener('load', checkIfInView);
    
    // Inicializar tooltips para enlaces de recursos
    const resourceLinks = document.querySelectorAll('.resource-link');
    
    resourceLinks.forEach(link => {
        link.setAttribute('title', 'Haz clic para ver más información');
    });
    
    // Validación de formularios (si existen)
    const forms = document.querySelectorAll('form');
    
    forms.forEach(form => {
        form.addEventListener('submit', function(e) {
            let isValid = true;
            const requiredFields = form.querySelectorAll('[required]');
            
            requiredFields.forEach(field => {
                if (!field.value.trim()) {
                    isValid = false;
                    field.classList.add('error');
                } else {
                    field.classList.remove('error');
                }
            });
            
            if (!isValid) {
                e.preventDefault();
                alert('Por favor, completa todos los campos requeridos.');
            }
        });
    });
    
    // Contador de progreso del curso (simulado)
    const progressElement = document.querySelector('.progress-fill');
    
    if (progressElement) {
        // Simular progreso basado en localStorage (en una implementación real, esto vendría del backend)
        let progress = localStorage.getItem('courseProgress') || 0;
        progressElement.style.width = progress + '%';
        
        // Actualizar progreso al hacer clic en enlaces de módulos (simulación)
        const moduleLinks = document.querySelectorAll('.module-btn');
        
        moduleLinks.forEach((link, index) => {
            link.addEventListener('click', function() {
                // Simular progreso basado en el módulo visitado
                const newProgress = Math.min(Math.round(((index + 1) / moduleLinks.length) * 100), 100);
                localStorage.setItem('courseProgress', newProgress);
            });
        });
    }
    
    // Verificar enlaces a recursos interactivos
    const interactiveLinks = document.querySelectorAll('.interactive-btn');
    
    interactiveLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            const href = this.getAttribute('href');
            
            // Verificar si el recurso existe (en una implementación real, esto sería más robusto)
            if (href.includes('../interactivos/')) {
                console.log('Accediendo a recurso interactivo:', href);
                // Aquí se podría implementar una verificación real de la existencia del recurso
            }
        });
    });
    
    // Inicialización de la página
    console.log('Curso Interactivo de Arquitecturas CNN cargado correctamente');
});

// Añadir estilos para animaciones
document.head.insertAdjacentHTML('beforeend', `
<style>
.animate-on-scroll {
    opacity: 0;
    transform: translateY(30px);
    transition: opacity 0.6s ease, transform 0.6s ease;
}
.animate-on-scroll.animate {
    opacity: 1;
    transform: translateY(0);
}
</style>
`);
