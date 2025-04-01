// Funcionalidad específica para las páginas de módulos
document.addEventListener('DOMContentLoaded', function() {
    // Navegación dentro del módulo
    const lessonLinks = document.querySelectorAll('.module-outline a');
    const lessons = document.querySelectorAll('.lesson');
    
    // Función para activar la lección correspondiente
    function activateLesson(targetId) {
        // Desactivar todos los enlaces y lecciones
        lessonLinks.forEach(link => link.classList.remove('active'));
        
        // Activar el enlace correspondiente
        const activeLink = document.querySelector(`.module-outline a[href="#${targetId}"]`);
        if (activeLink) {
            activeLink.classList.add('active');
        }
        
        // Actualizar URL con el hash
        history.pushState(null, null, `#${targetId}`);
    }
    
    // Configurar eventos para los enlaces de lecciones
    lessonLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            
            const targetId = this.getAttribute('href').substring(1);
            activateLesson(targetId);
            
            // Desplazarse a la lección
            const targetElement = document.getElementById(targetId);
            if (targetElement) {
                window.scrollTo({
                    top: targetElement.offsetTop - 150,
                    behavior: 'smooth'
                });
            }
        });
    });
    
    // Detectar la lección visible durante el scroll
    function highlightVisibleLesson() {
        let currentLessonId = null;
        let smallestDistance = Infinity;
        
        lessons.forEach(lesson => {
            const rect = lesson.getBoundingClientRect();
            const distance = Math.abs(rect.top - 150);
            
            if (distance < smallestDistance) {
                smallestDistance = distance;
                currentLessonId = lesson.id;
            }
        });
        
        if (currentLessonId) {
            activateLesson(currentLessonId);
        }
    }
    
    window.addEventListener('scroll', highlightVisibleLesson);
    
    // Inicializar con la lección actual basada en el hash de la URL
    function initializeFromHash() {
        const hash = window.location.hash.substring(1);
        if (hash && document.getElementById(hash)) {
            activateLesson(hash);
            
            // Desplazarse a la lección después de un breve retraso
            setTimeout(() => {
                const targetElement = document.getElementById(hash);
                window.scrollTo({
                    top: targetElement.offsetTop - 150,
                    behavior: 'auto'
                });
            }, 100);
        } else if (lessons.length > 0) {
            // Si no hay hash, activar la primera lección
            activateLesson(lessons[0].id);
        }
    }
    
    initializeFromHash();
    
    // Gestionar el progreso del módulo
    const completeButton = document.getElementById('mark-complete');
    const progressFill = document.querySelector('.progress-fill');
    const progressText = document.querySelector('.module-progress span');
    
    if (completeButton && progressFill && progressText) {
        // Verificar si el módulo ya está completado
        const moduleId = window.location.pathname.split('/').pop().replace('.html', '');
        const isCompleted = localStorage.getItem(`module_${moduleId}_completed`) === 'true';
        
        if (isCompleted) {
            progressFill.style.width = '100%';
            progressText.textContent = 'Progreso: 100%';
            completeButton.textContent = 'Módulo completado';
            completeButton.disabled = true;
        }
        
        // Configurar evento para marcar como completado
        completeButton.addEventListener('click', function() {
            // Animar la barra de progreso
            progressFill.style.width = '100%';
            progressText.textContent = 'Progreso: 100%';
            
            // Guardar estado en localStorage
            localStorage.setItem(`module_${moduleId}_completed`, 'true');
            
            // Actualizar botón
            completeButton.textContent = 'Módulo completado';
            completeButton.disabled = true;
            
            // Actualizar progreso general del curso
            updateCourseProgress();
        });
    }
    
    // Función para actualizar el progreso general del curso
    function updateCourseProgress() {
        // Obtener todos los módulos completados
        let completedModules = 0;
        const totalModules = 8; // Número total de módulos en el curso
        
        for (let i = 1; i <= totalModules; i++) {
            if (localStorage.getItem(`module_${i}_completed`) === 'true') {
                completedModules++;
            }
        }
        
        // Calcular y guardar el progreso general
        const overallProgress = Math.round((completedModules / totalModules) * 100);
        localStorage.setItem('courseProgress', overallProgress);
    }
    
    // Inicializar progreso de lectura
    function initializeReadingProgress() {
        const moduleId = window.location.pathname.split('/').pop().replace('.html', '');
        const lastPosition = localStorage.getItem(`module_${moduleId}_position`);
        
        if (lastPosition && !window.location.hash) {
            window.scrollTo({
                top: parseInt(lastPosition),
                behavior: 'auto'
            });
        }
        
        // Guardar posición de lectura periódicamente
        setInterval(() => {
            localStorage.setItem(`module_${moduleId}_position`, window.scrollY);
        }, 5000);
    }
    
    initializeReadingProgress();
    
    // Gestionar interacciones con recursos
    const resourceLinks = document.querySelectorAll('.module-resources a');
    
    resourceLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            // Registrar interacción con el recurso
            const resourceUrl = this.getAttribute('href');
            const resourceType = this.querySelector('i').className;
            
            console.log(`Recurso accedido: ${resourceUrl}, Tipo: ${resourceType}`);
            
            // Si es un recurso interactivo, verificar que existe
            if (resourceUrl.includes('../interactivos/')) {
                // En una implementación real, se verificaría la existencia del recurso
                console.log('Accediendo a recurso interactivo:', resourceUrl);
            }
        });
    });
});
