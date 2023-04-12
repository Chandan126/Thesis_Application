import { Component,OnInit } from '@angular/core';
import { DataServiceService } from '../data-service.service';
import { MatDialogRef } from '@angular/material/dialog';
import { Router } from '@angular/router';

@Component({
  selector: 'app-landing-popup',
  templateUrl: './landing-popup.component.html',
  styleUrls: ['./landing-popup.component.css']
})
export class LandingPopupComponent implements OnInit{
  constructor(private dialogRef: MatDialogRef<LandingPopupComponent>,private dataService: DataServiceService, private router: Router) {}

  ngOnInit() {
    console.log('Login Started');
  }

  login() {
    this.dataService.initSession();
    this.dialogRef.close();
  }

  logout() {
    this.dialogRef.close('logout'); // pass 'logout' as the result to trigger the redirection to login page
  }
}
