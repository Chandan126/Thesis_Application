import { ComponentFixture, TestBed } from '@angular/core/testing';

import { GlobalExplanationsComponent } from './global-explanations.component';

describe('GlobalExplanationsComponent', () => {
  let component: GlobalExplanationsComponent;
  let fixture: ComponentFixture<GlobalExplanationsComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ GlobalExplanationsComponent ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(GlobalExplanationsComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
