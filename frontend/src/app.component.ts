
import { Component, inject, OnInit, HostListener } from '@angular/core';
import { CommonModule } from '@angular/common';
import { VellumiumService } from './services/vellumium.service';
import { AuthService } from './services/auth.service';
import { LandingComponent } from './components/landing.component';
import { DashboardComponent } from './components/dashboard.component';
import { EditorComponent } from './components/editor.component';
import { StoryboardCanvasComponent } from './components/storyboard/storyboard-canvas.component';
import { DirectorsCutVideogenComponent } from './components/directors-cut/directorscut-videogen.component';
import { PremiereComponent } from './components/premiere.component';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [
    CommonModule,
    LandingComponent,
    DashboardComponent,
    EditorComponent,
    StoryboardCanvasComponent,
    DirectorsCutVideogenComponent,
    PremiereComponent
  ],
  template: `
    <main class="w-full min-h-screen bg-[--cinema-bg] relative text-[--cinema-text]">
      @switch (vellumService.currentView()) {
        @case ('LANDING') {
          <app-landing />
        }
        @case ('DASHBOARD') {
          <app-dashboard />
        }
        @case ('EDITOR') {
          <app-editor />
        }
        @case ('STORYBOARD_CANVAS') {
          <app-storyboard-canvas />
        }
        @case ('DIRECTORSCUT_VIDEOGEN') {
          <app-directorscut-videogen />
        }
        @case ('PREMIERE') {
          <app-premiere />
        }
      }

      <!-- Modern Transition Overlay -->
      <div 
        class="modern-transition" 
        [class.active]="vellumService.transitioning() !== 'idle'"
      ></div>
    </main>
  `
})
export class AppComponent implements OnInit {
  vellumService = inject(VellumiumService);
  authService = inject(AuthService);

  @HostListener('document:keydown.escape', ['$event'])
  handleEsc(event: KeyboardEvent) {
    const view = this.vellumService.currentView();
    if (view === 'STORYBOARD_CANVAS' || view === 'DIRECTORSCUT_VIDEOGEN') {
      this.vellumService.backToEditor();
    } else if (view === 'EDITOR' || view === 'PREMIERE') {
      this.vellumService.backToDashboard();
    }
  }

  ngOnInit() {
    setTimeout(() => {
      // Auto-navigate to dashboard if logged in (skipping landing)
      if (this.authService.isLoggedIn()) {
        if (this.vellumService.currentView() === 'LANDING') {
          this.vellumService.navigateTo('DASHBOARD', true);
        }
      }
    }, 100);
  }
}
