import { ComponentFixture, TestBed } from '@angular/core/testing';

import { LandingPopupComponent } from './landing-popup.component';

describe('LandingPopupComponent', () => {
  let component: LandingPopupComponent;
  let fixture: ComponentFixture<LandingPopupComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ LandingPopupComponent ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(LandingPopupComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
