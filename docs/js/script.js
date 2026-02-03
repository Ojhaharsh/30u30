/**
 * 30u30 Website - Interactive Features
 * Smooth animations, filtering, and user interactions
 */

document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    initScrollEffects();
    initPaperFilters();
    initCounterAnimation();
    initParticles();
    initFadeAnimations();
    initLiveStats();
});

/**
 * Navigation
 */
function initNavigation() {
    const nav = document.querySelector('.nav');
    const navToggle = document.querySelector('.nav-toggle');
    const navLinks = document.querySelector('.nav-links');
    
    // Scroll effect
    let lastScroll = 0;
    
    window.addEventListener('scroll', () => {
        const currentScroll = window.pageYOffset;
        
        if (currentScroll > 50) {
            nav.classList.add('scrolled');
        } else {
            nav.classList.remove('scrolled');
        }
        
        lastScroll = currentScroll;
    });
    
    // Mobile menu toggle
    if (navToggle) {
        navToggle.addEventListener('click', () => {
            navToggle.classList.toggle('open');
            navLinks.classList.toggle('open');
            document.body.style.overflow = navLinks.classList.contains('open') ? 'hidden' : '';
        });
        
        // Close menu on link click
        navLinks.querySelectorAll('a').forEach(link => {
            link.addEventListener('click', () => {
                navToggle.classList.remove('open');
                navLinks.classList.remove('open');
                document.body.style.overflow = '';
            });
        });
    }
    
    // Smooth scroll for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                const navHeight = nav.offsetHeight;
                const targetPosition = target.offsetTop - navHeight - 20;
                
                window.scrollTo({
                    top: targetPosition,
                    behavior: 'smooth'
                });
            }
        });
    });
}

/**
 * Scroll-based Effects
 */
function initScrollEffects() {
    // Parallax for hero symbols
    const symbols = document.querySelectorAll('.symbol');
    
    window.addEventListener('scroll', () => {
        const scrolled = window.pageYOffset;
        
        symbols.forEach((symbol, index) => {
            const speed = 0.1 + (index * 0.05);
            symbol.style.transform = `translateY(${scrolled * speed}px)`;
        });
    });
}

/**
 * Paper Filtering
 */
function initPaperFilters() {
    const filterBtns = document.querySelectorAll('.filter-btn');
    const paperCards = document.querySelectorAll('.paper-card');
    
    if (!filterBtns.length) return;
    
    filterBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            // Update active button
            filterBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            const filter = btn.dataset.filter;
            
            // Filter cards with animation
            paperCards.forEach((card, index) => {
                const unit = card.dataset.unit;
                const shouldShow = filter === 'all' || unit === filter;
                
                if (shouldShow) {
                    card.style.display = '';
                    card.style.animation = `fadeInUp 0.4s ease ${index * 0.05}s both`;
                } else {
                    card.style.display = 'none';
                }
            });
        });
    });
}

/**
 * Counter Animation
 */
function initCounterAnimation() {
    const counters = document.querySelectorAll('[data-count]');
    
    if (!counters.length) return;
    
    const animateCounter = (counter) => {
        const target = parseInt(counter.dataset.count);
        const duration = 2000;
        const step = target / (duration / 16);
        let current = 0;
        
        const updateCounter = () => {
            current += step;
            if (current < target) {
                counter.textContent = Math.floor(current);
                requestAnimationFrame(updateCounter);
            } else {
                counter.textContent = target;
            }
        };
        
        updateCounter();
    };
    
    // Use Intersection Observer to trigger animation
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                animateCounter(entry.target);
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.5 });
    
    counters.forEach(counter => observer.observe(counter));
}

/**
 * Particle Effect for Hero
 */
function initParticles() {
    const particlesContainer = document.getElementById('particles');
    if (!particlesContainer) return;
    
    const particleCount = 30;
    
    for (let i = 0; i < particleCount; i++) {
        const particle = document.createElement('div');
        particle.className = 'particle';
        
        const size = Math.random() * 4 + 2;
        const posX = Math.random() * 100;
        const posY = Math.random() * 100;
        const delay = Math.random() * 5;
        const duration = Math.random() * 10 + 10;
        const opacity = Math.random() * 0.3 + 0.1;
        
        particle.style.cssText = `
            position: absolute;
            width: ${size}px;
            height: ${size}px;
            background: var(--color-primary);
            border-radius: 50%;
            left: ${posX}%;
            top: ${posY}%;
            opacity: ${opacity};
            animation: floatParticle ${duration}s ease-in-out ${delay}s infinite;
        `;
        
        particlesContainer.appendChild(particle);
    }
    
    // Add keyframe animation
    const style = document.createElement('style');
    style.textContent = `
        @keyframes floatParticle {
            0%, 100% {
                transform: translateY(0) translateX(0);
                opacity: var(--opacity, 0.2);
            }
            25% {
                transform: translateY(-30px) translateX(10px);
            }
            50% {
                transform: translateY(-10px) translateX(-10px);
                opacity: calc(var(--opacity, 0.2) * 0.5);
            }
            75% {
                transform: translateY(-40px) translateX(5px);
            }
        }
    `;
    document.head.appendChild(style);
}

/**
 * Fade-in Animations on Scroll
 */
function initFadeAnimations() {
    // Add fade-in class to elements
    const animatedElements = document.querySelectorAll(
        '.paper-card, .timeline-unit, .feature, .about-visual'
    );
    
    animatedElements.forEach(el => {
        el.classList.add('fade-in');
    });
    
    // Intersection Observer for fade-in
    const fadeObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
                fadeObserver.unobserve(entry.target);
            }
        });
    }, {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    });
    
    document.querySelectorAll('.fade-in').forEach(el => {
        fadeObserver.observe(el);
    });
}

/**
 * Utility: Debounce function
 */
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

/**
 * Paper Card Hover Effects
 */
document.querySelectorAll('.paper-card').forEach(card => {
    card.addEventListener('mouseenter', function(e) {
        // Add subtle glow effect on hover
        this.style.setProperty('--mouse-x', e.offsetX + 'px');
        this.style.setProperty('--mouse-y', e.offsetY + 'px');
    });
    
    card.addEventListener('mousemove', function(e) {
        this.style.setProperty('--mouse-x', e.offsetX + 'px');
        this.style.setProperty('--mouse-y', e.offsetY + 'px');
    });
});

/**
 * Keyboard Navigation Support
 */
document.addEventListener('keydown', (e) => {
    // Close mobile menu on Escape
    if (e.key === 'Escape') {
        const navToggle = document.querySelector('.nav-toggle');
        const navLinks = document.querySelector('.nav-links');
        
        if (navLinks && navLinks.classList.contains('open')) {
            navToggle.classList.remove('open');
            navLinks.classList.remove('open');
            document.body.style.overflow = '';
        }
    }
});

/**
 * Performance: Lazy load images if any
 */
if ('loading' in HTMLImageElement.prototype) {
    // Native lazy loading supported
    document.querySelectorAll('img[data-src]').forEach(img => {
        img.src = img.dataset.src;
    });
} else {
    // Fallback for older browsers
    const lazyImages = document.querySelectorAll('img[data-src]');
    
    const imageObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const img = entry.target;
                img.src = img.dataset.src;
                imageObserver.unobserve(img);
            }
        });
    });
    
    lazyImages.forEach(img => imageObserver.observe(img));
}

/**
 * Live Stats - GoatCounter Integration
 */
function initLiveStats() {
    const statsSection = document.getElementById('stats');
    if (!statsSection) return;
    
    // Animate stats counters on scroll into view
    const statNumbers = statsSection.querySelectorAll('.stat-number');
    
    const animateStat = (element, target, suffix = '') => {
        const duration = 2500;
        const steps = 60;
        const stepValue = target / steps;
        let current = 0;
        let step = 0;
        
        const animate = () => {
            step++;
            // Ease out effect
            const progress = step / steps;
            const easeOut = 1 - Math.pow(1 - progress, 3);
            current = Math.floor(target * easeOut);
            
            // Format number with commas for large numbers
            const formatted = current >= 1000 ? 
                current.toLocaleString() : current;
            
            element.innerHTML = formatted + suffix;
            
            if (step < steps) {
                requestAnimationFrame(animate);
            } else {
                element.innerHTML = target.toLocaleString() + suffix;
            }
        };
        
        animate();
    };
    
    // Observer for stats section
    const statsObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                // Animate each stat with slight delay
                statNumbers.forEach((stat, index) => {
                    setTimeout(() => {
                        const target = parseInt(stat.dataset.target) || 0;
                        const suffix = stat.dataset.suffix || '';
                        animateStat(stat, target, suffix);
                    }, index * 150);
                });
                
                // Add hotspot animations
                initMapHotspots();
                
                statsObserver.unobserve(entry.target);
            }
        });
    }, { threshold: 0.2 });
    
    statsObserver.observe(statsSection);
    
    // Try to fetch real stats from GoatCounter if available
    fetchGoatCounterStats();
}

/**
 * Fetch stats from GoatCounter API
 */
async function fetchGoatCounterStats() {
    // Note: GoatCounter public API requires authentication
    // This is a placeholder for when the user sets up their account
    // The initial values are placeholder data
    
    try {
        // Check if GoatCounter is loaded
        if (window.goatcounter && window.goatcounter.count) {
            console.log('📊 GoatCounter is tracking visits');
        }
    } catch (e) {
        console.log('📊 Using placeholder stats - set up GoatCounter for real data');
    }
}

/**
 * Initialize map hotspot animations
 */
function initMapHotspots() {
    const mapWrapper = document.querySelector('.world-map-wrapper');
    if (!mapWrapper) return;
    
    // Sample hotspot locations (percentage based)
    const hotspots = [
        { x: 25, y: 35, label: 'USA' },      // North America
        { x: 48, y: 30, label: 'Europe' },   // Europe
        { x: 75, y: 40, label: 'India' },    // India
        { x: 82, y: 55, label: 'Australia' }, // Australia
        { x: 55, y: 35, label: 'UAE' },      // Middle East
        { x: 85, y: 35, label: 'Japan' },    // Japan
    ];
    
    hotspots.forEach((spot, index) => {
        setTimeout(() => {
            const hotspot = document.createElement('div');
            hotspot.className = 'map-hotspot';
            hotspot.style.left = spot.x + '%';
            hotspot.style.top = spot.y + '%';
            hotspot.title = spot.label;
            
            // Add tooltip on hover
            hotspot.setAttribute('data-tooltip', spot.label);
            
            mapWrapper.appendChild(hotspot);
        }, index * 200);
    });
}

/**
 * Country bar animation
 */
document.querySelectorAll('.country-bar-fill').forEach((bar, index) => {
    const width = bar.style.width || bar.dataset.width || '50%';
    bar.style.width = '0%';
    
    setTimeout(() => {
        bar.style.transition = 'width 1s ease-out';
        bar.style.width = width;
    }, 500 + (index * 100));
});

console.log('🚀 30u30 website initialized');
