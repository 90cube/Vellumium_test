
import { Component, inject, effect, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { LocationService } from '../../services/location.service';
import { VellumiumService } from '../../services/vellumium.service';
import { Location, LOCATION_SLIDER_DEFS } from '../../models/location.model';

@Component({
  selector: 'app-location-panel',
  standalone: true,
  imports: [CommonModule, FormsModule],
  template: `
    <div class="flex flex-col h-full">

      <!-- Header -->
      <div class="flex items-center justify-between px-4 py-3 border-b border-[--cinema-border]">
        <h3 class="text-xs font-display font-semibold text-[--cinema-text-muted] uppercase tracking-wider">Locations</h3>
        <button
          (click)="addLocation()"
          class="p-1 rounded hover:bg-[--cinema-primary]/10 text-[--cinema-text-muted] hover:text-[--cinema-primary] transition-colors"
          title="Add Location">
          <span class="material-symbols-outlined text-[16px]">add</span>
        </button>
      </div>

      <!-- Location List -->
      <div class="overflow-y-auto p-3 border-b border-[--cinema-border]" style="max-height: 220px;">
        @if (locationService.locations().length === 0) {
          <div class="text-xs text-[--cinema-text-dim] text-center py-6 font-display">No locations yet</div>
        } @else {
          <div class="grid grid-cols-3 gap-2">
            @for (loc of locationService.locations(); track loc.id) {
              <div
                draggable="true"
                (dragstart)="onDragStart($event, loc)"
                class="rounded-lg border cursor-pointer transition-all overflow-hidden"
                [class]="locationService.selectedLocation()?.id === loc.id
                  ? 'border-[--cinema-primary]/60 bg-[--cinema-primary]/5'
                  : 'border-[--cinema-border] bg-[--cinema-surface] hover:border-[--cinema-primary]/20'"
                (click)="selectLocation(loc)">
                <!-- Thumbnail / Placeholder -->
                <div class="w-full aspect-square flex items-center justify-center bg-[--cinema-bg]">
                  @if (loc.thumbnail_url) {
                    <img [src]="loc.thumbnail_url" class="w-full h-full object-cover" alt="">
                  } @else {
                    <span class="material-symbols-outlined text-[24px] text-[--cinema-text-dim]">terrain</span>
                  }
                </div>
                <!-- Name -->
                <div class="px-1.5 py-1 text-center">
                  <span class="text-[10px] text-[--cinema-text] truncate block">{{ loc.name }}</span>
                </div>
              </div>
            }
          </div>
        }
      </div>

      <!-- Editor (when location selected) -->
      @if (locationService.selectedLocation(); as loc) {
        <div class="flex-1 overflow-y-auto p-4 space-y-4">

          <!-- Name -->
          <div class="space-y-1">
            <label class="text-[10px] font-display font-semibold text-[--cinema-text-muted] uppercase tracking-wider">Name</label>
            <input
              type="text"
              [ngModel]="loc.name"
              (ngModelChange)="onNameChange($event)"
              class="w-full bg-[--cinema-bg] text-[--cinema-text] text-xs border border-[--cinema-border] rounded-lg px-3 py-1.5 outline-none focus:border-[--cinema-primary]/30">
          </div>

          <!-- Sliders -->
          <div class="space-y-2">
            <label class="text-[10px] font-display font-semibold text-[--cinema-text-muted] uppercase tracking-wider">Environment</label>
            @for (slider of sliderDefs; track slider.id) {
              <div class="space-y-0.5">
                <div class="flex items-center justify-between">
                  <span class="text-[10px] text-[--cinema-text]">{{ slider.label }}</span>
                  <span class="text-[10px] font-mono w-10 text-right"
                    [class]="getSliderValue(loc, slider.id) === 0
                      ? 'text-[--cinema-text-dim]'
                      : 'text-[--cinema-primary]'">{{ getSliderValue(loc, slider.id).toFixed(2) }}</span>
                </div>
                <div class="flex items-center gap-1.5">
                  <span class="text-[9px] text-[--cinema-text-dim] w-10 text-right shrink-0">{{ slider.negLabel }}</span>
                  <input type="range" min="-200" max="200" step="1"
                    [value]="getSliderValue(loc, slider.id) * 100"
                    (input)="onSliderChange(slider.id, +$any($event.target).value / 100)"
                    (dblclick)="onSliderChange(slider.id, 0)"
                    class="flex-1 h-1 accent-[--cinema-primary] cursor-pointer">
                  <span class="text-[9px] text-[--cinema-text-dim] w-10 shrink-0">{{ slider.posLabel }}</span>
                </div>
              </div>
            }
          </div>

          <!-- Prompt -->
          <div class="space-y-1">
            <label class="text-[10px] font-display font-semibold text-[--cinema-text-muted] uppercase tracking-wider">Prompt</label>
            <textarea
              [ngModel]="loc.prompt"
              (ngModelChange)="onPromptChange($event)"
              class="w-full h-16 bg-[--cinema-bg] text-[--cinema-text] text-xs border border-[--cinema-border] rounded-lg px-3 py-2 focus:ring-1 focus:ring-[--cinema-primary]/30 focus:border-[--cinema-primary]/50 focus:outline-none resize-none"
              placeholder="Location-specific prompt keywords..."></textarea>
          </div>

          <!-- Actions -->
          <div class="flex gap-2">
            <button
              (click)="saveLocation()"
              class="flex-1 py-1.5 text-[11px] font-display font-semibold rounded-lg transition-all
                bg-gradient-to-r from-[--cinema-primary] to-[--cinema-accent] text-white
                hover:shadow-lg hover:shadow-[--cinema-primary]/20">
              Save
            </button>
            <button
              (click)="deleteLocation()"
              class="px-3 py-1.5 text-[11px] rounded-lg border border-[--cinema-border] text-[--cinema-text-muted]
                hover:text-[--cinema-error] hover:border-[--cinema-error]/30 transition-colors">
              <span class="material-symbols-outlined text-[14px]">delete</span>
            </button>
          </div>
        </div>
      }

    </div>
  `
})
export class LocationPanelComponent implements OnInit {
  locationService = inject(LocationService);
  vellumiumService = inject(VellumiumService);

  sliderDefs = LOCATION_SLIDER_DEFS;

  // Local edit state (buffered changes before save)
  private editName = '';
  private editSliders: Record<string, number> = {};
  private editPrompt = '';

  constructor() {
    // Reload locations when project changes
    effect(() => {
      const project = this.vellumiumService.selectedProject();
      if (project) {
        this.locationService.loadLocations(project.id);
        this.locationService.selectedLocation.set(null);
      }
    });
  }

  ngOnInit() {
    const project = this.vellumiumService.selectedProject();
    if (project) {
      this.locationService.loadLocations(project.id);
    }
  }

  selectLocation(loc: Location) {
    this.locationService.selectedLocation.set(loc);
    this.editName = loc.name;
    this.editSliders = { ...loc.sliders };
    this.editPrompt = loc.prompt;
  }

  getSliderValue(loc: Location, sliderId: string): number {
    // Use buffered edit value if this is the selected location
    const selected = this.locationService.selectedLocation();
    if (selected && selected.id === loc.id && sliderId in this.editSliders) {
      return this.editSliders[sliderId] ?? 0;
    }
    return loc.sliders[sliderId] ?? 0;
  }

  onNameChange(value: string) {
    this.editName = value;
  }

  onSliderChange(sliderId: string, value: number) {
    this.editSliders[sliderId] = value;
    // Update the selected location signal for immediate UI feedback
    const selected = this.locationService.selectedLocation();
    if (selected) {
      this.locationService.selectedLocation.set({
        ...selected,
        sliders: { ...this.editSliders },
      });
    }
  }

  onPromptChange(value: string) {
    this.editPrompt = value;
  }

  async addLocation() {
    const project = this.vellumiumService.selectedProject();
    if (!project) return;

    const count = this.locationService.locations().length;
    const name = 'Location ' + (count + 1);
    const created = await this.locationService.createLocation(project.id, name);
    if (created) {
      this.selectLocation(created);
    }
  }

  async saveLocation() {
    const selected = this.locationService.selectedLocation();
    if (!selected) return;

    await this.locationService.updateLocation(selected.id, {
      name: this.editName,
      sliders: { ...this.editSliders },
      prompt: this.editPrompt,
    });
  }

  onDragStart(event: DragEvent, loc: Location) {
    event.dataTransfer?.setData('application/json', JSON.stringify({ type: 'location', id: loc.id }));
    event.dataTransfer!.effectAllowed = 'link';
  }

  async deleteLocation() {
    const selected = this.locationService.selectedLocation();
    if (!selected) return;

    await this.locationService.deleteLocation(selected.id);
  }
}
