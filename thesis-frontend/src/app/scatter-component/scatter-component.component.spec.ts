import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ScatterComponentComponent } from './scatter-component.component';

describe('ScatterComponentComponent', () => {
  let component: ScatterComponentComponent;
  let fixture: ComponentFixture<ScatterComponentComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ ScatterComponentComponent ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(ScatterComponentComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
