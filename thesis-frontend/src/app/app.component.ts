import { Component, OnInit } from '@angular/core';
import {Router} from '@angular/router'
import { MatDialog } from '@angular/material/dialog';
import { DataServiceService } from '../app/data-service.service';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit {
  title = 'thesis-frontend';
  sessionId: any;

  constructor(private dialog: MatDialog, private router: Router, private dataService: DataServiceService){
  }

  ngOnInit() {
    const sessionId = localStorage.getItem('sessionId');
    if (sessionId) {
      console.log(`Session already started ${sessionId}`);
      this.router.navigateByUrl('/home');
    } else {
      this.router.navigateByUrl('/login');
    }
  }

  logout() {
    this.dataService.logout().subscribe();
    if(localStorage.getItem('sessionId')!==undefined){
      localStorage.removeItem('sessionId');
    }
    console.log('Session removed');
    this.router.navigateByUrl('/login');
  }
}
