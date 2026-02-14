
import { Component, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { VellumiumService } from './services/vellumium.service';
import { LandingComponent } from './components/landing.component';
import { DashboardComponent } from './components/dashboard.component';
import { EditorComponent } from './components/editor.component';
import { CanvasComponent } from './components/canvas.component';
import { PremiereComponent } from './components/premiere.component';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [
    CommonModule, 
    LandingComponent, 
    DashboardComponent, 
    EditorComponent, 
    CanvasComponent, 
    PremiereComponent
  ],
  template: `
    <main class="w-full min-h-screen bg-black">
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
        @case ('CANVAS') {
          <app-canvas />
        }
        @case ('PREMIERE') {
          <app-premiere />
        }
      }
    </main>
  `
})
export class AppComponent {
  vellumService = inject(VellumiumService);
}
