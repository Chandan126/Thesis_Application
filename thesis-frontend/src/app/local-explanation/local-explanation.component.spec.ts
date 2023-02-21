import { ComponentFixture, TestBed } from '@angular/core/testing';

import { LocalExplanationComponent } from './local-explanation.component';

describe('LocalExplanationComponent', () => {
  let component: LocalExplanationComponent;
  let fixture: ComponentFixture<LocalExplanationComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ LocalExplanationComponent ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(LocalExplanationComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
