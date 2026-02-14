
import { Component, ElementRef, signal, inject, OnInit, OnDestroy, ChangeDetectionStrategy } from '@angular/core';

@Component({
  selector: 'app-robot-mascot',
  standalone: true,
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div class="booth-container relative" #boothContainer>
      <!-- SVG Booth -->
      <svg viewBox="0 0 400 500" class="booth-svg relative z-10" xmlns="http://www.w3.org/2000/svg">
        <defs>
          <!-- Booth Wood Gradients -->
          <linearGradient id="woodDark" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stop-color="#3e2723"/>
            <stop offset="50%" stop-color="#5d4037"/>
            <stop offset="100%" stop-color="#3e2723"/>
          </linearGradient>
          <linearGradient id="woodLight" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stop-color="#8d6e63"/>
            <stop offset="100%" stop-color="#6d4c41"/>
          </linearGradient>
          
          <!-- Metal Gradients -->
          <linearGradient id="gold" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%" stop-color="#ffd54f"/>
            <stop offset="50%" stop-color="#ffecb3"/>
            <stop offset="100%" stop-color="#ffca28"/>
          </linearGradient>
          
          <!-- Robot Materials (Reused) -->
          <linearGradient id="robotBody" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%" stop-color="#d4956b"/>
            <stop offset="100%" stop-color="#a86438"/>
          </linearGradient>

          <!-- Glass Reflection -->
          <linearGradient id="glassReflect" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%" stop-color="white" stop-opacity="0.1"/>
            <stop offset="40%" stop-color="white" stop-opacity="0.3"/>
            <stop offset="60%" stop-color="white" stop-opacity="0"/>
          </linearGradient>

          <filter id="boothShadow">
            <feGaussianBlur in="SourceAlpha" stdDeviation="6"/>
            <feOffset dx="4" dy="8"/>
            <feComponentTransfer>
              <feFuncA type="linear" slope="0.4"/>
            </feComponentTransfer>
            <feMerge>
              <feMergeNode/>
              <feMergeNode in="SourceGraphic"/>
            </feMerge>
          </filter>

          <filter id="neonGlow">
            <feGaussianBlur stdDeviation="2.5" result="coloredBlur"/>
            <feMerge>
              <feMergeNode in="coloredBlur"/>
              <feMergeNode in="SourceGraphic"/>
            </feMerge>
          </filter>
        </defs>

        <!-- ====== BOOTH STRUCTURE ====== -->
        <g filter="url(#boothShadow)">
          <!-- Base Platform -->
          <rect x="20" y="440" width="360" height="40" rx="4" fill="url(#woodDark)" stroke="#2d1b15" stroke-width="2"/>
          
          <!-- Main Cabinet Box -->
          <rect x="40" y="60" width="320" height="400" rx="8" fill="#2d1b15"/>
           <!-- Inner Background (Curtain/Void) -->
          <rect x="60" y="80" width="280" height="280" fill="#1a1a2e"/>
          
          <!-- Decorative Pillars -->
          <rect x="40" y="60" width="25" height="400" fill="url(#woodLight)" stroke="#3e2723"/>
          <rect x="335" y="60" width="25" height="400" fill="url(#woodLight)" stroke="#3e2723"/>
          
          <!-- Top Ornament Header -->
          <path d="M30,60 L370,60 L360,20 L40,20 Z" fill="url(#woodDark)" stroke="#2d1b15"/>
          <rect x="50" y="30" width="300" height="20" rx="2" fill="#1a1a2e"/>
          <!-- Sign Text -->
          <text x="200" y="45" text-anchor="middle" font-family="serif" font-weight="bold" font-size="20" fill="#ffd54f" letter-spacing="6" filter="url(#neonGlow)">VELLUMIUM</text>

          <!-- Ticket Slot Area (Bottom) -->
          <rect x="65" y="370" width="270" height="90" rx="4" fill="url(#woodLight)" stroke="#3e2723"/>
          
          <!-- Control Panel Inset -->
          <rect x="80" y="385" width="240" height="60" rx="4" fill="#2d1b15" stroke="#1a1a2e" stroke-width="2"/>
        </g>

        <!-- ====== ROBOT (Inside Booth) ====== -->
        <g transform="translate(0, 20)"> <!-- Moved up slightly to fit booth -->
           <!-- Clip path to keep robot inside booth window -->
           <clipPath id="boothWindow">
             <rect x="60" y="80" width="280" height="280"/>
           </clipPath>

           <g clip-path="url(#boothWindow)">
              <!-- Body (Static) -->
              <rect x="120" y="240" width="160" height="140" rx="16" fill="url(#robotBody)" stroke="#5d4037" stroke-width="2"/>
              <!-- Neck -->
              <rect x="185" y="220" width="30" height="25" fill="#5d4037"/>

              <!-- HEAD (Follows Mouse) -->
              <g class="robot-head"
                 [style.transform]="'translate(' + headX() + 'px, ' + headY() + 'px) rotate(' + headRotate() + 'deg)'"
                 style="transform-origin: 200px 170px; transition: transform 0.1s ease-out;">
                
                <!-- Antenna -->
                <line x1="200" y1="110" x2="200" y2="80" stroke="#5d4037" stroke-width="4"/>
                <circle cx="200" cy="80" r="6" fill="#7a1ff9" filter="url(#neonGlow)">
                   <animate attributeName="opacity" values="1;0.5;1" dur="2s" repeatCount="indefinite"/>
                </circle>

                <!-- Head Box -->
                <rect x="135" y="110" width="130" height="120" rx="12" fill="url(#robotBody)" stroke="#5d4037" stroke-width="2"/>
                
                <!-- Face Screen -->
                <rect x="150" y="130" width="100" height="80" rx="8" fill="#1a1a2e"/>

                <!-- EYES -->
                <g class="eyes">
                  @if (!isBlinking()) {
                    <circle cx="175" cy="165" r="10" fill="#00f2ff" filter="url(#neonGlow)" opacity="0.9"/>
                    <circle cx="225" cy="165" r="10" fill="#00f2ff" filter="url(#neonGlow)" opacity="0.9"/>
                  } @else {
                    <!-- Blinking Line -->
                    <line x1="165" y1="165" x2="185" y2="165" stroke="#00f2ff" stroke-width="2" opacity="0.8"/>
                    <line x1="215" y1="165" x2="235" y2="165" stroke="#00f2ff" stroke-width="2" opacity="0.8"/>
                  }
                </g>

                <!-- MOUTH (Simple Line or Talking) -->
                 @switch (mouthState()) {
                   @case ('smile') {
                     <path d="M180,195 Q200,205 220,195" fill="none" stroke="#7a1ff9" stroke-width="3" stroke-linecap="round" opacity="0.8"/>
                   }
                   @case ('o') {
                     <circle cx="200" cy="195" r="4" fill="none" stroke="#7a1ff9" stroke-width="3" opacity="0.8"/>
                   }
                   @default {
                     <rect x="180" y="190" width="40" height="4" rx="2" fill="#7a1ff9" opacity="0.7"/>
                   }
                 }
              </g>

              <!-- Hands (Resting on counter) -->
              <circle cx="100" cy="350" r="20" fill="#d4956b" stroke="#5d4037" stroke-width="2"/>
              <circle cx="300" cy="350" r="20" fill="#d4956b" stroke="#5d4037" stroke-width="2"/>
           </g>
        </g>

        <!-- ====== GLASS PANE (Overlay) ====== -->
        <path d="M60,80 L340,80 L340,360 L60,360 Z" fill="url(#glassReflect)" style="pointer-events: none;"/>
        <!-- Reflection Streak -->
        <path d="M60,360 L160,80 L200,80 L100,360 Z" fill="white" opacity="0.05" style="pointer-events: none;"/>

      </svg>
      
      <!-- CLOUD / THOUGHT BUBBLE (IDLE ANIMATION) -->
      @if (showThought()) {
        <g class="thought-bubble" transform="translate(250, 40)" filter="url(#boothShadow)" style="opacity: 0; animation: fadeIn 0.5s forwards;">
           <svg x="200" y="-80" width="200" height="200" viewBox="0 0 200 200" style="overflow: visible;">
             <!-- Bubble Shape -->
             <path d="M100,100 Q80,90 90,70 Q80,40 110,30 Q140,10 180,30 Q220,40 210,70 Q230,100 190,110 Q170,130 130,120 Q110,130 90,100 Z" 
                   fill="white" stroke="#2d1b15" stroke-width="2" opacity="0.9"/>
             <!-- Small circles leading from robot head (around x=200, y=110) -->
             <circle cx="130" cy="140" r="8" fill="white" stroke="#2d1b15" stroke-width="1.5"/>
             <circle cx="120" cy="160" r="5" fill="white" stroke="#2d1b15" stroke-width="1.5"/>

             <!-- CONTENT: PROJECTOR -->
             @if (thoughtContent() === 'projector') {
               <g transform="translate(130, 60) scale(0.6)">
                 <rect x="0" y="0" width="60" height="40" rx="2" fill="#333"/>
                 <circle cx="30" cy="20" r="15" fill="#444" stroke="#666">
                   <animateTransform attributeName="transform" type="rotate" from="0 30 20" to="360 30 20" dur="2s" repeatCount="indefinite"/>
                 </circle>
                 <circle cx="10" cy="-15" r="12" fill="none" stroke="#222" stroke-width="3">
                   <animateTransform attributeName="transform" type="rotate" from="0 10 -15" to="360 10 -15" dur="3s" repeatCount="indefinite"/>
                 </circle>
                 <circle cx="50" cy="-15" r="12" fill="none" stroke="#222" stroke-width="3">
                   <animateTransform attributeName="transform" type="rotate" from="360 50 -15" to="0 50 -15" dur="3s" repeatCount="indefinite"/>
                 </circle>
                 <path d="M60,10 L100, -20 L100, 60 Z" fill="rgba(255,255,255,0.3)">
                    <animate attributeName="opacity" values="0.1;0.4;0.1" dur="0.5s" repeatCount="indefinite"/>
                 </path>
               </g>
             }

             <!-- CONTENT: PALETTE -->
             @if (thoughtContent() === 'palette') {
               <g transform="translate(130, 60) scale(0.6)">
                 <path d="M10,0 Q-10,10 0,30 Q10,50 40,40 Q60,30 50,10 Q40,-10 10,0 Z" fill="#d4a373" stroke="#8d6e63"/>
                 <circle cx="10" cy="15" r="3" fill="#ef4444"/>
                 <circle cx="20" cy="25" r="3" fill="#3b82f6"/>
                 <circle cx="35" cy="20" r="3" fill="#22c55e"/>
                 <!-- Brush -->
                 <rect x="40" y="-10" width="5" height="40" fill="#8d6e63" transform="rotate(45 40 20)">
                    <animateTransform attributeName="transform" type="translate" values="0,0; 5,-5; 0,0" dur="1s" repeatCount="indefinite"/>
                 </rect>
               </g>
             }
           </svg>
        </g>
      }

      <!-- Overlay for buttons (positioned precisely over the Control Panel Inset) 
           SVG viewBox is 0 0 400 500. 
           Control Panel Inset is roughly y=385 to y=445. 
           385/500 = 77% from top. 
           Height 60/500 = 12%.
      -->
      <div class="absolute top-[77%] left-0 right-0 flex justify-center items-center gap-2 z-50 px-12 h-[12%] pointer-events-auto">
        <ng-content></ng-content>
      </div>
    </div>
  `,
  styles: [`
    :host {
      display: block;
      width: 100%;
      max-width: 400px;
      margin: 0 auto;
      /* Allow pointer events on the host so children can receive them, 
         but disable on the SVG container to prevent blocking */
    }
    .booth-container {
      position: relative;
      width: 100%;
    }
    .booth-svg {
      width: 100%;
      height: auto;
      display: block;
      filter: drop-shadow(0 10px 20px rgba(0,0,0,0.5));
      pointer-events: none; /* SVG shouldn't block clicks */
    }
    .pointer-events-auto {
      pointer-events: auto;
    }
    .robot-head {
      will-change: transform;
    }
    @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
  `]
})
export class RobotMascotComponent implements OnInit, OnDestroy {
  private el = inject(ElementRef);

  headX = signal(0);
  headY = signal(0);
  headRotate = signal(0);

  // Easter Egg Signals
  isBlinking = signal(false);
  mouthState = signal<'flat' | 'smile' | 'o'>('flat');
  showThought = signal(false);
  thoughtContent = signal<'projector' | 'palette'>('projector');

  private mouseMoveHandler: ((e: MouseEvent) => void) | null = null;
  private animFrameId = 0;

  // Timers
  private blinkTimeout: any;
  private mouthTimeout: any;
  private idleTimeout: any;

  // Movement State
  private targetX = 0;
  private targetY = 0;
  private targetR = 0;
  private currentX = 0;
  private currentY = 0;
  private currentR = 0;

  ngOnInit() {
    this.mouseMoveHandler = (e: MouseEvent) => {
      this.resetIdleTimer(); // Activity detected

      const rect = this.el.nativeElement.getBoundingClientRect();
      const centerX = rect.left + rect.width / 2;
      const centerY = rect.top + rect.height * 0.4; // Aim for head area

      const dx = e.clientX - centerX;
      const dy = e.clientY - centerY;

      // Restrict movement to look like it's inside the booth
      const maxMove = 15;
      const maxRotate = 8;

      // Calculate normalized distance
      const dist = Math.sqrt(dx * dx + dy * dy);
      const normFactor = Math.min(dist, 600) / 600;

      this.targetX = (dx / 600) * maxMove;
      this.targetY = (dy / 600) * maxMove * 0.8;
      this.targetR = (dx / 600) * maxRotate;
    };

    window.addEventListener('mousemove', this.mouseMoveHandler);
    this.animate();

    // Start Random loops
    this.scheduleBlink();
    this.scheduleMouth();
    this.resetIdleTimer();
  }

  private scheduleBlink() {
    const nextBlink = Math.random() * 3000 + 2000; // 2-5s
    this.blinkTimeout = setTimeout(() => {
      this.isBlinking.set(true);
      setTimeout(() => this.isBlinking.set(false), 150);
      this.scheduleBlink();
    }, nextBlink);
  }

  private scheduleMouth() {
    const nextChange = Math.random() * 4000 + 3000; // 3-7s
    this.mouthTimeout = setTimeout(() => {
      const states: ('flat' | 'smile' | 'o')[] = ['flat', 'smile', 'o', 'flat'];
      const pick = states[Math.floor(Math.random() * states.length)];
      this.mouthState.set(pick);
      this.scheduleMouth();
    }, nextChange);
  }

  private resetIdleTimer() {
    this.showThought.set(false);
    if (this.idleTimeout) clearTimeout(this.idleTimeout);

    // If idle for 4 seconds, show thought
    this.idleTimeout = setTimeout(() => {
      this.thoughtContent.set(Math.random() > 0.5 ? 'projector' : 'palette');
      this.showThought.set(true);
    }, 4000);
  }

  private animate() {
    const lerp = 0.1; // Slightly snappier for mechanical feel
    this.currentX += (this.targetX - this.currentX) * lerp;
    this.currentY += (this.targetY - this.currentY) * lerp;
    this.currentR += (this.targetR - this.currentR) * lerp;

    const rx = Math.round(this.currentX * 100) / 100;
    const ry = Math.round(this.currentY * 100) / 100;
    const rr = Math.round(this.currentR * 100) / 100;

    if (Math.abs(rx - this.headX()) > 0.01 ||
      Math.abs(ry - this.headY()) > 0.01 ||
      Math.abs(rr - this.headRotate()) > 0.01) {
      this.headX.set(rx);
      this.headY.set(ry);
      this.headRotate.set(rr);
    }
    this.animFrameId = requestAnimationFrame(() => this.animate());
  }

  ngOnDestroy() {
    if (this.mouseMoveHandler) {
      window.removeEventListener('mousemove', this.mouseMoveHandler);
    }
    if (this.animFrameId) cancelAnimationFrame(this.animFrameId);
    if (this.blinkTimeout) clearTimeout(this.blinkTimeout);
    if (this.mouthTimeout) clearTimeout(this.mouthTimeout);
    if (this.idleTimeout) clearTimeout(this.idleTimeout);
  }
}
