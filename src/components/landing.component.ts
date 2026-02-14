
import { Component, signal, inject } from '@angular/core';
import { CommonModule, NgOptimizedImage } from '@angular/common';
import { VellumiumService } from '../services/vellumium.service';

@Component({
  selector: 'app-landing',
  standalone: true,
  imports: [CommonModule, NgOptimizedImage],
  template: `
    <div class="relative w-full bg-zinc-950 overflow-hidden text-zinc-200 selection:bg-amber-900 selection:text-white">
      
      <!-- The "Film Gate" (Landing Hero) -->
      <!-- Fixed container that slides open when signing in -->
      <div class="fixed inset-0 z-50 flex pointer-events-none">
        
        <!-- Left Curtain (Film Strip) -->
        <div 
          class="h-full bg-black border-r-4 border-zinc-900 transition-transform duration-[1200ms] ease-[cubic-bezier(0.65,0,0.35,1)] flex flex-col relative overflow-hidden shadow-2xl"
          [class.-translate-x-full]="animateOut()"
          [style.width.%]="50"
        >
           <!-- Film Sprockets Left -->
           <div class="absolute left-2 top-0 bottom-0 w-8 flex flex-col gap-6 py-4">
             @for(i of sprockets; track i) { <div class="w-6 h-4 bg-zinc-800/50 rounded-sm shadow-inner"></div> }
           </div>
           <!-- Film Texture -->
           <div class="absolute inset-0 bg-[url('https://www.transparenttextures.com/patterns/stardust.png')] opacity-20"></div>
           
           <!-- Content on Left Curtain -->
           <div class="absolute inset-0 flex items-center justify-end pr-10 opacity-30 pointer-events-none">
              <span class="font-cinema text-9xl text-zinc-900 rotate-90 transform origin-right">SCENE 1</span>
           </div>
        </div>

        <!-- Right Curtain (Film Strip) -->
        <div 
          class="h-full bg-black border-l-4 border-zinc-900 transition-transform duration-[1200ms] ease-[cubic-bezier(0.65,0,0.35,1)] flex flex-col relative overflow-hidden shadow-2xl"
          [class.translate-x-full]="animateOut()"
          [style.width.%]="50"
        >
           <!-- Film Sprockets Right -->
           <div class="absolute right-2 top-0 bottom-0 w-8 flex flex-col gap-6 py-4">
              @for(i of sprockets; track i) { <div class="w-6 h-4 bg-zinc-800/50 rounded-sm shadow-inner"></div> }
           </div>
           <!-- Film Texture -->
           <div class="absolute inset-0 bg-[url('https://www.transparenttextures.com/patterns/stardust.png')] opacity-20"></div>

           <!-- Content on Right Curtain -->
           <div class="absolute inset-0 flex items-center justify-start pl-10 opacity-30 pointer-events-none">
              <span class="font-cinema text-9xl text-zinc-900 -rotate-90 transform origin-left">TAKE 1</span>
           </div>
        </div>

        <!-- Center Stage (Logo & Auth) - Attached to the gate visually but fades out or scales -->
        <div class="absolute inset-0 flex flex-col items-center justify-center pointer-events-auto transition-all duration-700"
             [class.opacity-0]="animateOut()"
             [class.scale-150]="animateOut()">
          
          <div class="relative z-10 p-12 bg-black/40 backdrop-blur-sm border border-zinc-800/50 rounded-lg shadow-2xl text-center transform transition-transform hover:scale-105 duration-500">
             <!-- Decorative Corners -->
             <div class="absolute top-0 left-0 w-4 h-4 border-t-2 border-l-2 border-amber-500"></div>
             <div class="absolute top-0 right-0 w-4 h-4 border-t-2 border-r-2 border-amber-500"></div>
             <div class="absolute bottom-0 left-0 w-4 h-4 border-b-2 border-l-2 border-amber-500"></div>
             <div class="absolute bottom-0 right-0 w-4 h-4 border-b-2 border-r-2 border-amber-500"></div>

             <h1 class="font-cinema text-6xl md:text-8xl text-transparent bg-clip-text bg-gradient-to-b from-amber-300 to-amber-700 tracking-widest drop-shadow-sm mb-4">
               VELLUMIUM
             </h1>
             <p class="font-script text-zinc-400 text-sm tracking-[0.3em] uppercase mb-8">
               EST. 2024 • The Director's Suite
             </p>

             <button 
                (click)="onSignIn()"
                class="group relative inline-flex items-center justify-center px-8 py-3 overflow-hidden font-cinema font-bold text-white transition-all duration-300 bg-amber-900/20 border border-amber-600 rounded-sm hover:bg-amber-600 hover:border-amber-500 focus:outline-none focus:ring-2 focus:ring-amber-400 focus:ring-offset-2 focus:ring-offset-gray-900">
                <span class="absolute w-0 h-0 transition-all duration-500 ease-out bg-amber-600 rounded-full group-hover:w-56 group-hover:h-56 opacity-10"></span>
                <span class="relative">ADMIT ONE • SIGN IN</span>
             </button>
          </div>
          
          <div class="absolute bottom-12 animate-bounce flex flex-col items-center gap-2 opacity-50">
            <span class="text-amber-500 font-cinema text-xl">↓</span>
            <span class="font-script text-[10px] text-zinc-500 tracking-widest uppercase">Scroll for Roadmap</span>
          </div>

        </div>
      </div>

      <!-- Scrollable Roadmap Content (Visible behind/below the gate conceptually) -->
      <!-- We add padding-top 100vh so it appears 'below' the initial view -->
      <div class="relative z-10 w-full min-h-screen pt-[100vh] flex flex-col items-center bg-zinc-950">
        
        <div class="w-full max-w-5xl px-6 py-24 space-y-24">
          
          <div class="text-center space-y-4">
             <h2 class="font-cinema text-4xl text-amber-500">The Production Pipeline</h2>
             <div class="w-24 h-1 bg-amber-800 mx-auto"></div>
             <p class="font-script text-zinc-400 max-w-2xl mx-auto">
               From the first spark of an idea to the silver screen premiere. Vellumium manages the entire lifecycle of your cinematic masterpiece.
             </p>
          </div>

          <!-- Roadmap Steps -->
          <div class="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div class="group p-8 border border-zinc-800 bg-zinc-900/30 hover:bg-zinc-900 hover:border-amber-700 transition-colors duration-300 relative overflow-hidden">
               <div class="absolute top-0 right-0 p-4 font-cinema text-6xl text-zinc-800 group-hover:text-amber-900/50 transition-colors">01</div>
               <h3 class="font-cinema text-2xl text-amber-100 mb-4 relative z-10">Pre-Production</h3>
               <ul class="font-script text-sm text-zinc-400 space-y-2 relative z-10 list-disc pl-4">
                 <li>AI-Assisted Storyboarding</li>
                 <li>Script Generation</li>
                 <li>Character Design</li>
               </ul>
            </div>

            <div class="group p-8 border border-zinc-800 bg-zinc-900/30 hover:bg-zinc-900 hover:border-amber-700 transition-colors duration-300 relative overflow-hidden">
               <div class="absolute top-0 right-0 p-4 font-cinema text-6xl text-zinc-800 group-hover:text-amber-900/50 transition-colors">02</div>
               <h3 class="font-cinema text-2xl text-amber-100 mb-4 relative z-10">Production</h3>
               <ul class="font-script text-sm text-zinc-400 space-y-2 relative z-10 list-disc pl-4">
                 <li>Scene Assembly</li>
                 <li>Layer-based Canvas</li>
                 <li>GenAI Video Rendering</li>
               </ul>
            </div>

            <div class="group p-8 border border-zinc-800 bg-zinc-900/30 hover:bg-zinc-900 hover:border-amber-700 transition-colors duration-300 relative overflow-hidden">
               <div class="absolute top-0 right-0 p-4 font-cinema text-6xl text-zinc-800 group-hover:text-amber-900/50 transition-colors">03</div>
               <h3 class="font-cinema text-2xl text-amber-100 mb-4 relative z-10">Premiere</h3>
               <ul class="font-script text-sm text-zinc-400 space-y-2 relative z-10 list-disc pl-4">
                 <li>Community Theater</li>
                 <li>Director's Cut Screenings</li>
                 <li>Critique & Acclaim</li>
               </ul>
            </div>
          </div>

          <!-- Footer -->
          <footer class="text-center font-script text-xs text-zinc-600 pt-12 border-t border-zinc-900">
             &copy; 2024 Vellumium Archives. All Rights Reserved.
          </footer>

        </div>
      </div>

    </div>
  `
})
export class LandingComponent {
  vellumService = inject(VellumiumService);
  animateOut = signal(false);
  sprockets = Array(40).fill(0).map((_, i) => i); // Generate enough sprocket holes

  onSignIn() {
    this.animateOut.set(true);
    this.vellumService.login();
    
    // Duration matches the CSS transition time
    setTimeout(() => {
      this.vellumService.navigateTo('DASHBOARD');
    }, 1200);
  }
}
