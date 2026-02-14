
import { Component, signal, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { VellumiumService } from '../services/vellumium.service';
import { AuthService } from '../services/auth.service';
import { RobotMascotComponent } from './robot-mascot.component';

@Component({
  selector: 'app-landing',
  standalone: true,
  imports: [CommonModule, FormsModule, RobotMascotComponent],
  template: `
    <div class="relative w-full min-h-screen bg-[var(--cinema-bg)] text-[var(--cinema-text)] font-sans overflow-x-hidden selection:bg-[var(--cinema-primary)] selection:text-white">

      <!-- Background Gradients -->
      <div class="fixed inset-0 pointer-events-none">
        <div class="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-[var(--cinema-primary)] opacity-[0.08] blur-[120px] rounded-full"></div>
        <div class="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-[var(--cinema-accent)] opacity-[0.05] blur-[120px] rounded-full"></div>
      </div>

      <!-- Navigation -->
      <nav class="fixed top-0 w-full z-50 px-6 py-4 flex items-center justify-between backdrop-blur-md bg-[var(--cinema-bg)]/50 border-b border-[var(--cinema-border)]">
        <div class="flex items-center">
          <img src="assets/logo/small-white-logo.png" alt="Vellumium" class="h-8 w-auto object-contain opacity-90" />
        </div>
        <div>
          @if (!authService.isLoggedIn()) {
            <button (click)="showAuthModal(false)"
              class="px-5 py-2 text-sm font-medium text-[var(--cinema-text-muted)] hover:text-white transition-colors">
              Sign In
            </button>
            <button (click)="showAuthModal(true)"
              class="ml-4 px-5 py-2 text-sm font-medium bg-[var(--cinema-primary)] hover:bg-[#6015c7] text-white rounded-lg transition-all shadow-lg shadow-[var(--cinema-primary-dim)]">
              Get Started
            </button>
          } @else {
            <button (click)="authService.signOut()"
              class="px-4 py-2 text-sm font-medium text-[var(--cinema-text-muted)] hover:text-[#ef4444] transition-colors mr-2">
              Logout
            </button>
            <button (click)="enterApp()"
              class="px-6 py-2 text-sm font-medium bg-[var(--cinema-primary)] hover:bg-[#6015c7] text-white rounded-lg transition-all shadow-lg shadow-[var(--cinema-primary-dim)]">
              Launch Studio
            </button>
          }
        </div>
      </nav>

      <!-- HERO SECTION -->
      <section class="relative pt-24 pb-20 px-6 flex flex-col items-center text-center z-10 min-h-screen justify-center">
        
        <!-- 1. Robot Booth (Hero Centerpiece) -->
        <div class="mb-12 scale-110 md:scale-125 animate-fade-in-up">
          <app-robot-mascot>
            <!-- Actions Projected into Booth Counter -->
            @if (authService.isLoggedIn()) {
              <button (click)="enterApp()" class="group relative px-6 py-2 bg-[var(--cinema-primary)] hover:bg-[#6015c7] text-white text-xs font-bold rounded overflow-hidden shadow-[0_0_15px_var(--cinema-primary-dim)] transition-all transform hover:scale-105 hover:shadow-[0_0_25px_var(--cinema-primary)] border border-white/20">
                <div class="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-700"></div>
                <div class="flex items-center gap-2">
                  <span class="material-symbols-outlined text-sm">rocket_launch</span>
                  <span class="tracking-widest">LAUNCH STUDIO</span>
                </div>
              </button>
            } @else {
              <!-- Ticket Button -->
              <button (click)="showAuthModal(false)" class="group relative px-4 py-2 bg-[#2d1b15] hover:bg-[#3e2723] text-[#ffd54f] border border-[#ffd54f]/30 text-[10px] font-bold rounded shadow-lg uppercase tracking-widest transition-all hover:border-[#ffd54f] flex items-center gap-1">
                 <span class="material-symbols-outlined text-[14px]">confirmation_number</span>
                 Check Ticket
              </button>
              
              <!-- Issue Button -->
              <button (click)="showAuthModal(true)" class="group relative px-4 py-2 bg-[#1a1a2e] hover:bg-[#252540] text-[var(--cinema-accent)] border border-[var(--cinema-accent)]/30 text-[10px] font-bold rounded shadow-lg uppercase tracking-widest transition-all hover:border-[var(--cinema-accent)] flex items-center gap-1">
                 <span class="material-symbols-outlined text-[14px]">local_activity</span>
                 Get Ticket
              </button>
            }
          </app-robot-mascot>
        </div>

        <!-- 2. Main Text (Moved Below) -->
        <div class="max-w-4xl mx-auto flex flex-col items-center animate-fade-in-up delay-200">
          <div class="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-[var(--cinema-panel)] border border-[var(--cinema-border)] mb-6">
            <span class="w-2 h-2 rounded-full bg-[var(--cinema-accent)] animate-pulse"></span>
            <span class="text-xs font-medium text-[var(--cinema-accent)] tracking-wide uppercase">AI Video Production V2.0</span>
          </div>

          <h1 class="text-4xl md:text-6xl font-bold tracking-tight mb-6 bg-clip-text text-transparent bg-gradient-to-b from-white to-[var(--cinema-text-muted)]">
            Direct Your Imagination
          </h1>
          
          <p class="text-lg text-[var(--cinema-text-muted)] max-w-2xl leading-relaxed mb-10">
            The all-in-one AI film studio. Storyboard, generate, and edit professional video content in a unified, professional workspace.
          </p>

          <button (click)="scrollToFeatures()" class="text-[var(--cinema-text-dim)] hover:text-white transition-colors flex flex-col items-center gap-2 text-sm uppercase tracking-widest">
            Explore Features
            <span class="material-symbols-outlined animate-bounce">arrow_downward</span>
          </button>
        </div>

      </section>

      <!-- FEATURE GRID -->
      <section id="features" class="py-32 px-6 relative z-10">
        <div class="max-w-6xl mx-auto">
          <div class="text-center mb-20">
            <h2 class="text-3xl md:text-5xl font-bold mb-6">Complete Production Pipeline</h2>
            <p class="text-[var(--cinema-text-muted)] max-w-2xl mx-auto">
              From script to final render. Vellumium replaces the fragmented AI toolchain with a single, cohesive workflow.
            </p>
          </div>

          <div class="grid grid-cols-1 md:grid-cols-3 gap-8">
            <!-- Feature 1 -->
            <div class="glass-panel p-8 hover:bg-white/5 transition-all group">
              <div class="w-12 h-12 rounded-lg bg-[var(--cinema-surface)] flex items-center justify-center border border-[var(--cinema-border)] mb-6 group-hover:border-[var(--cinema-primary)] transition-colors">
                <span class="material-symbols-outlined text-[var(--cinema-primary)]">style</span>
              </div>
              <h3 class="text-xl font-bold mb-3">AI Storyboarding</h3>
              <p class="text-[var(--cinema-text-muted)] leading-relaxed">
                Visualize scenes instantly. Use ControlNet for precise composition and LoRAs for consistent style across frames.
              </p>
            </div>

            <!-- Feature 2 -->
            <div class="glass-panel p-8 hover:bg-white/5 transition-all group">
              <div class="w-12 h-12 rounded-lg bg-[var(--cinema-surface)] flex items-center justify-center border border-[var(--cinema-border)] mb-6 group-hover:border-[var(--cinema-accent)] transition-colors">
                <span class="material-symbols-outlined text-[var(--cinema-accent)]">movie_filter</span>
              </div>
              <h3 class="text-xl font-bold mb-3">Motion Generation</h3>
              <p class="text-[var(--cinema-text-muted)] leading-relaxed">
                Turn static frames into dynamic video clips. Control camera movement (pan, tilt, zoom) with simple directives.
              </p>
            </div>

            <!-- Feature 3 -->
            <div class="glass-panel p-8 hover:bg-white/5 transition-all group">
              <div class="w-12 h-12 rounded-lg bg-[var(--cinema-surface)] flex items-center justify-center border border-[var(--cinema-border)] mb-6 group-hover:border-[#ec4899] transition-colors">
                <span class="material-symbols-outlined text-[#ec4899]">auto_fix_high</span>
              </div>
              <h3 class="text-xl font-bold mb-3">Professional NLE</h3>
              <p class="text-[var(--cinema-text-muted)] leading-relaxed">
                A full non-linear editor in your browser. Multi-track timeline, transitions, audio mixing, and 4K export.
              </p>
            </div>
          </div>
        </div>
      </section>

      <!-- TICKET AUTH COMPONENT (IMAGE BASED) -->
      @if (showModal()) {
        <div class="fixed inset-0 z-[100] flex items-center justify-center bg-[var(--cinema-surface)]/90 backdrop-blur-sm animate-fade-in"
          (mousedown)="onMousedown($event)" (mouseup)="onMouseup($event)">
          
          <!-- Ticket Container -->
          <div class="ticket-container scale-100 animate-scale-in flex items-center justify-center gap-[2px]" (click)="$event.stopPropagation()">
            
            <!-- LEFT PART (ID/PW Image) -->
            <div class="relative group">
               <!-- Image Asset: Always use Light Mode (as requested) -->
               <img src="assets/ticket/IDPW_light_mode.png"
                    alt="Ticket ID/PW" 
                    class="h-[200px] w-auto object-contain drop-shadow-2xl select-none" 
                    draggable="false">

               <!-- Inputs Overlay -->
               <div class="absolute inset-0 flex flex-col items-center justify-center pb-2">
                   <!-- Header Overlay (Logo Centered) -->
                   <div class="w-full flex justify-center items-center px-5 mb-3">
                      <img src="assets/ticket/logo_white.png" 
                           alt="Vellumium" 
                           class="h-4 w-auto object-contain select-none opacity-90" 
                           draggable="false">
                   </div>
                   
                   <!-- ID Input Area -->
                   <div class="w-full px-[36px] mt-[4px] flex items-end gap-2">
                      <span class="text-[#ffecb3] font-bold text-xs mb-1 opacity-80 font-mono w-[22px]">ID:</span>
                      <input [(ngModel)]="email" placeholder="director@example.com" 
                             class="flex-1 bg-transparent border-b border-[#ffecb3]/50 focus:border-[#ffecb3] focus:outline-none text-xs font-bold tracking-wider font-mono placeholder-[#ffecb3]/30 text-[#ffecb3] pb-1 transition-colors h-5"
                             style="text-shadow: 0 1px 1px rgba(0,0,0,0.1); caret-color: #ffecb3;"/>
                   </div>

                   <!-- PW Input Area -->
                   <div class="w-full px-[36px] mt-[14px] flex items-end gap-2 relative">
                       <span class="text-[#ffecb3] font-bold text-xs mb-1 opacity-80 font-mono w-[22px]">PW:</span>
                       <div class="flex-1 relative">
                         <input [type]="showPassword() ? 'text' : 'password'" [(ngModel)]="password" placeholder="••••••••" (keydown.enter)="onTicketLogin()"
                               class="w-full bg-transparent border-b border-[#ffecb3]/50 focus:border-[#ffecb3] focus:outline-none text-xs font-bold tracking-wider font-mono placeholder-[#ffecb3]/30 text-[#ffecb3] pb-1 pr-6 transition-colors h-5"
                               style="caret-color: #ffecb3;"/>
                         
                         <button (click)="showPassword.set(!showPassword())" 
                                 class="absolute right-0 top-1/2 -translate-y-1/2 opacity-50 hover:opacity-100 text-[#ffecb3] transition-opacity p-1">
                           <span class="material-symbols-outlined text-[10px]">{{ showPassword() ? 'visibility' : 'visibility_off' }}</span>
                         </button>
                       </div>
                   </div>
               </div>
            </div>

            <!-- RIGHT PART (LOGIN Image Button) -->
            <div class="relative cursor-pointer hover:brightness-110 active:scale-[0.98] transition-all flex items-center justify-center" 
                 (click)="onTicketLogin()">
               <img src="assets/ticket/LOGIN_light_mode.png"
                    alt="Login Ticket" 
                    class="h-[200px] w-auto object-contain drop-shadow-2xl select-none" 
                    draggable="false">
               
               <!-- Log In Text Overlay -->
               <span class="absolute text-[#ffecb3] font-bold tracking-widest leading-none select-none pointer-events-none transform rotate-180"
                     style="writing-mode: vertical-rl; text-orientation: mixed; font-size: 1.1rem; opacity: 0.9;">
                   LOG IN
               </span>
            </div>

          </div>
          
          <!-- Bottom Actions (Outside Ticket) -->
          <div class="absolute bottom-[18%] flex flex-col items-center gap-4 animate-fade-in delay-200">
             
             <!-- Google Login -->
             <button (click)="onGoogleSignIn()" 
                     class="flex items-center gap-3 px-6 py-2 rounded-full bg-[var(--cinema-panel)] border border-[var(--cinema-border)] hover:bg-[var(--cinema-border)] transition-all group backdrop-blur-md">
                <svg class="w-5 h-5 grayscale group-hover:grayscale-0 transition-all" viewBox="0 0 24 24"><path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92a5.06 5.06 0 0 1-2.2 3.32v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.1z" fill="#4285F4"/><path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853"/><path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" fill="#FBBC05"/><path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335"/></svg>
                <span class="text-sm font-medium text-[var(--cinema-text)]">Sign in with Google</span>
             </button>

             <!-- Toggle Login/Signup Mode -->
             <div class="text-[10px] uppercase tracking-widest cursor-pointer hover:underline text-[var(--cinema-text-muted)] font-bold opacity-80"
                  (click)="setMode(!isSignUpMode())">
               {{ isSignUpMode() ? 'Have a ticket?' : 'Need a ticket?' }}
             </div>

          </div>

        </div>
      }

      <!-- Footer -->
      <footer class="py-12 border-t border-[var(--cinema-border)] bg-[var(--cinema-bg)] text-center flex flex-col items-center gap-4">
        <img src="assets/logo/small-white-logo.png" alt="Vellumium" class="h-6 w-auto opacity-50 shadow-lg" />
        <p class="text-[var(--cinema-text-dim)] text-sm">&copy; 2026 Vellumium AI. All rights reserved.</p>
      </footer>

    </div>
  `,
  styles: [`
    .animate-fade-in-up {
      animation: fadeInUp 0.8s ease-out forwards;
      opacity: 0;
      transform: translateY(20px);
    }
    
    .delay-100 { animation-delay: 0.1s; }
    .delay-200 { animation-delay: 0.2s; }
    .delay-300 { animation-delay: 0.3s; }
    .delay-500 { animation-delay: 0.5s; }

    @keyframes fadeInUp {
      to {
        opacity: 1;
        transform: translateY(0);
      }
    }

    .animate-scale-in {
      animation: scaleIn 0.3s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }

    @keyframes scaleIn {
      from { transform: scale(0.95); opacity: 0; }
      to { transform: scale(1); opacity: 1; }
    }
    
    .animate-fade-in {
      animation: fadeIn 0.2s ease-out forwards;
    }

    @keyframes fadeIn {
      from { opacity: 0; }
      to { opacity: 1; }
    }
    
    /* Input Overlay Colors */
    :host {
      --ticket-text-overlay: #2d1b15; /* Default for Light Mode (Black Ticket?? No wait.) */
    }
  `]
})
export class LandingComponent {
  vellumService = inject(VellumiumService);
  authService = inject(AuthService);

  animateOut = signal(false);
  showModal = signal(false);
  isSignUpMode = signal(false);
  showPassword = signal(false);
  email = '';
  password = '';

  // Ticket Animation & Theme
  isTearing = signal(false);
  theme = signal<'dark' | 'light'>('dark');

  constructor() {
    // Initial check for overlay color based on default dark theme
    this.updateOverlayColor();
  }

  showAuthModal(isSignUp: boolean) {
    this.isSignUpMode.set(isSignUp);
    this.showModal.set(true);
    this.isTearing.set(false);
  }

  enterApp() {
    this.vellumService.navigateTo('DASHBOARD', true);
  }

  scrollToFeatures() {
    document.getElementById('features')?.scrollIntoView({ behavior: 'smooth' });
  }

  setMode(isSignUp: boolean) {
    this.isSignUpMode.set(isSignUp);
    this.authService.authError.set(null);
  }

  closeModal() {
    this.showModal.set(false);
    this.authService.authError.set(null);
    this.showPassword.set(false);
    this.isTearing.set(false);
  }

  toggleTheme() {
    const newTheme = this.theme() === 'dark' ? 'light' : 'dark';
    this.theme.set(newTheme);
    document.documentElement.setAttribute('data-theme', newTheme);
    document.body.setAttribute('data-theme', newTheme);
    this.updateOverlayColor();
  }

  updateOverlayColor() {
    // If Theme is Dark -> Gold Ticket -> Text should be Dark/Black? Or Gold?
    // Users Image: Dark Background -> Gold Ticket -> Black Text?

    // Let's assume:
    // Dark Mode (Gold Ticket) -> Text color: #2d1b15 (Dark Brown/Black)
    // Light Mode (Black Ticket) -> Text color: #ffd700 (Gold) or #ffffff (White)

    const root = document.documentElement;
    if (this.theme() === 'dark') {
      root.style.setProperty('--ticket-text-overlay', '#3e2723'); // Dark Brown on Gold
    } else {
      root.style.setProperty('--ticket-text-overlay', '#ffecb3'); // Light Gold on Black
    }
  }

  private isMouseDownOnOverlay = false;

  onMousedown(event: MouseEvent) {
    this.isMouseDownOnOverlay = event.target === event.currentTarget;
  }

  onMouseup(event: MouseEvent) {
    if (this.isMouseDownOnOverlay && event.target === event.currentTarget) {
      this.closeModal();
    }
    this.isMouseDownOnOverlay = false;
  }

  async onTicketLogin() {
    if (this.isTearing() || this.authService.authLoading()) return;

    if (!this.email.trim() || !this.password.trim()) {
      this.authService.authError.set('Please fill out the ticket.');
      return;
    }

    // Attempt Login/Signup
    let success: boolean;
    if (this.isSignUpMode()) {
      success = await this.authService.signUpWithEmail(this.email, this.password);
    } else {
      success = await this.authService.signInWithEmail(this.email, this.password);
    }

    // Play Animation on Success
    if (success) {
      this.showModal.set(false);
      this.enterApp();
    }
  }

  async onGoogleSignIn() {
    await this.authService.signInWithGoogle();
  }
}
